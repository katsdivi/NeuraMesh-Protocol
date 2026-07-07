//
//  Bonjour.swift
//  NMP — Phase 4
//
//  Bonjour/mDNS service publishing and browsing for zero-configuration
//  peer discovery on the local network.
//
//  Every device publishes `NeuraMesh-{peerID-hex}._neuramesh._tcp.local.`
//  with its capability advertisement in the TXT record, and browses for the
//  same type. Capabilities therefore propagate BEFORE any NMP connection
//  exists — the election can run on TXT data alone, and NMP handshakes are
//  only dialed where the coordinator decides they are needed (Phase 5).
//
//  The service type is `_tcp` and the publisher owns a real TCP listener:
//  Network.framework ties Bonjour registration lifetime to a listener, and
//  advertising over `_tcp` is the standard registration path (NMP data
//  itself stays UDP; the TCP port is a discovery anchor, not a data plane).
//  Inbound TCP connections are refused — Phase 4 is discovery only.
//
//  KNOWN LIMITATION (flagged in Phase4_Design.md): some managed networks
//  block mDNS entirely. Manual peer entry fallback is a Phase 6+ concern.
//

import Foundation
import Network

// MARK: - Naming

public enum NMPBonjour {
    public static let serviceType = "_neuramesh._tcp"
    public static let serviceDomain = "local."
    static let serviceNamePrefix = "NeuraMesh-"

    public static func serviceName(for peerID: UInt32) -> String {
        serviceNamePrefix + String(format: "%08x", peerID)
    }

    /// Recovers the peer ID from a service name. Needed on removal events,
    /// where the TXT record may no longer be available.
    public static func peerID(fromServiceName name: String) -> UInt32? {
        guard name.hasPrefix(serviceNamePrefix) else { return nil }
        return UInt32(name.dropFirst(serviceNamePrefix.count), radix: 16)
    }
}

public enum NMPBonjourError: Error, Equatable, Sendable {
    case invalidPort(UInt16)
    case alreadyStarted
}

// MARK: - Publisher

/// Publishes this device's NMP service + capability TXT record.
/// Conforms to `NMPCapabilityPublisher` so `NMPPeerDiscoveryManager` can
/// drive it (and tests can substitute a mock).
public final class NMPBonjourPublisher: NMPCapabilityPublisher {

    /// Diagnostics (listener state transitions), invoked on an internal queue.
    public var onStateChanged: ((NWListener.State) -> Void)?
    /// Port actually bound (useful when constructed with port 0 = ephemeral).
    public private(set) var listeningPort: UInt16?

    private let requestedPort: UInt16
    private var capabilities: NMPCapabilities
    private var listener: NWListener?
    private let queue = DispatchQueue(label: "nmp.bonjour.publisher")

    /// - Parameter port: NMP listening port to advertise; 0 = let the
    ///   system pick an ephemeral port.
    public init(capabilities: NMPCapabilities, port: UInt16) {
        self.capabilities = capabilities
        self.requestedPort = port
    }

    public func start() throws {
        guard listener == nil else { throw NMPBonjourError.alreadyStarted }
        let listener: NWListener
        if requestedPort == 0 {
            listener = try NWListener(using: .tcp)
        } else {
            guard let port = NWEndpoint.Port(rawValue: requestedPort) else {
                throw NMPBonjourError.invalidPort(requestedPort)
            }
            listener = try NWListener(using: .tcp, on: port)
        }
        listener.service = service(for: capabilities)
        // Discovery anchor only — refuse actual TCP connections.
        listener.newConnectionHandler = { $0.cancel() }
        listener.stateUpdateHandler = { [weak self] state in
            if case .ready = state {
                self?.listeningPort = self?.listener?.port?.rawValue
            }
            self?.onStateChanged?(state)
        }
        self.listener = listener
        listener.start(queue: queue)
    }

    /// Re-registers the service with a fresh TXT record. Browsers observe
    /// this as a `.changed` result with `.metadataChanged` set.
    public func update(capabilities: NMPCapabilities) {
        self.capabilities = capabilities
        listener?.service = service(for: capabilities)
    }

    public func stop() {
        listener?.cancel()
        listener = nil
        listeningPort = nil
    }

    private func service(for capabilities: NMPCapabilities) -> NWListener.Service {
        NWListener.Service(
            name: NMPBonjour.serviceName(for: capabilities.peerID),
            type: NMPBonjour.serviceType,
            domain: NMPBonjour.serviceDomain,
            txtRecord: NWTXTRecord(capabilities.txtDictionary())
        )
    }
}

// MARK: - Browser

/// Browses for NMP services and surfaces (capabilities, endpoint) pairs.
/// Conforms to `NMPPeerDiscoverySource` so `NMPPeerDiscoveryManager` can
/// consume it (and tests can substitute a deterministic mock).
///
/// NOTE: the browser reports EVERY matching service, including this
/// device's own — the manager filters self by peer ID.
public final class NMPBonjourBrowser: NMPPeerDiscoverySource {

    public var onPeerFound: ((NMPCapabilities, NWEndpoint?) -> Void)?
    public var onPeerLost: ((UInt32) -> Void)?
    /// Diagnostics (browser state transitions), invoked on an internal queue.
    public var onStateChanged: ((NWBrowser.State) -> Void)?

    private var browser: NWBrowser?
    private let queue = DispatchQueue(label: "nmp.bonjour.browser")

    public init() {}

    public func start() throws {
        guard browser == nil else { throw NMPBonjourError.alreadyStarted }
        // bonjourWithTXTRecord: TXT records arrive with browse results, so
        // capabilities are known without resolving or connecting.
        let descriptor = NWBrowser.Descriptor.bonjourWithTXTRecord(
            type: NMPBonjour.serviceType, domain: NMPBonjour.serviceDomain)
        let browser = NWBrowser(for: descriptor, using: .tcp)
        browser.browseResultsChangedHandler = { [weak self] _, changes in
            for change in changes {
                switch change {
                case .added(let result):
                    self?.announceFound(result)
                case .changed(_, let new, let flags) where flags.contains(.metadataChanged):
                    self?.announceFound(new) // capability update (TXT changed)
                case .removed(let result):
                    self?.announceLost(result)
                default:
                    break // interface changes, .identical
                }
            }
        }
        browser.stateUpdateHandler = { [weak self] state in
            self?.onStateChanged?(state)
        }
        self.browser = browser
        browser.start(queue: queue)
    }

    public func stop() {
        browser?.cancel()
        browser = nil
    }

    private func announceFound(_ result: NWBrowser.Result) {
        guard case .bonjour(let txt) = result.metadata,
              let capabilities = NMPCapabilities(txtDictionary: txt.dictionary) else {
            return // no/unparseable TXT — not a (compatible) NMP peer
        }
        onPeerFound?(capabilities, result.endpoint)
    }

    private func announceLost(_ result: NWBrowser.Result) {
        let peerID: UInt32?
        if case .service(let name, _, _, _) = result.endpoint {
            peerID = NMPBonjour.peerID(fromServiceName: name)
        } else if case .bonjour(let txt) = result.metadata,
                  let capabilities = NMPCapabilities(txtDictionary: txt.dictionary) {
            peerID = capabilities.peerID
        } else {
            peerID = nil
        }
        if let peerID { onPeerLost?(peerID) }
    }
}
