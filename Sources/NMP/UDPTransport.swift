//
//  UDPTransport.swift
//  NMP — Phase 1
//
//  Datagram transport abstraction + Network.framework UDP implementation.
//
//  The abstraction exists so PeerConnection can be driven by a deterministic
//  in-memory transport in tests (see Tests/NMPTests/MockTransport.swift) while
//  production uses NWConnection/NWListener. Per the spec's tech-stack rules,
//  no async/await: everything runs on caller-supplied dispatch queues.
//

import Foundation
import Network

// MARK: - Transport abstraction

/// What kind of physical path a transport runs over, as far as AWDL-style
/// airtime contention is concerned. Loopback and wired Ethernet cannot
/// experience Wi-Fi/AWDL contention, so traffic shaping is meaningless
/// there; `.unknown` is treated like `.radio` (shape conservatively).
public enum NMPLinkKind: Sendable {
    case wiredOrLoopback
    case radio
    case unknown
}

/// A bidirectional, unreliable, unordered datagram channel to ONE remote peer.
public protocol NMPTransport: AnyObject {
    /// Delivery callback. Invoked on the transport's queue with one complete
    /// datagram per call. Set before calling `start()`.
    var onReceive: ((Data) -> Void)? { get set }
    /// Invoked when the underlying channel dies (interface loss, etc.).
    var onClosed: ((Error?) -> Void)? { get set }
    /// Physical-path classification, used to gate AWDL contention shaping.
    /// May start `.unknown` and settle once the path is established.
    var linkKind: NMPLinkKind { get }
    /// Largest datagram this transport can actually put on the wire, if it
    /// knows (kernel/UDP limits — 9216 B on stock macOS). nil = unknown.
    var maxDatagramBytes: Int? { get }

    func start()
    func send(_ datagram: Data)
    func cancel()
    /// Runs `body`, coalescing the `send` calls made inside it where the
    /// transport can (NWConnection.batch amortizes per-send wakeups).
    func batched(_ body: () -> Void)
}

public extension NMPTransport {
    /// In-memory/test transports don't model an interface; shape as if radio.
    var linkKind: NMPLinkKind { .unknown }
    var maxDatagramBytes: Int? { nil }
    func batched(_ body: () -> Void) { body() }
}

// MARK: - Network.framework UDP client transport

/// Outbound UDP transport (initiator side, or responder replying to a known
/// endpoint). One NWConnection per remote peer.
public final class UDPTransport: NMPTransport {

    public var onReceive: ((Data) -> Void)?
    public var onClosed: ((Error?) -> Void)?
    public private(set) var linkKind: NMPLinkKind = .unknown
    public private(set) var maxDatagramBytes: Int?

    private let connection: NWConnection
    private let queue: DispatchQueue

    public init(host: NWEndpoint.Host, port: NWEndpoint.Port, queue: DispatchQueue) {
        let params = NWParameters.udp
        params.allowLocalEndpointReuse = true
        self.connection = NWConnection(host: host, port: port, using: params)
        self.queue = queue
    }

    /// Dial an arbitrary endpoint — notably a Bonjour `.service` endpoint
    /// from a browse result. Network.framework resolves the service's SRV
    /// record and connects UDP to the advertised port; NMP peers bind
    /// their UDP listener and their Bonjour TCP anchor to the SAME port
    /// number precisely so this resolution lands on the NMP listener
    /// (see PeerNode.swift).
    public init(endpoint: NWEndpoint, queue: DispatchQueue) {
        let params = NWParameters.udp
        params.allowLocalEndpointReuse = true
        self.connection = NWConnection(to: endpoint, using: params)
        self.queue = queue
    }

    /// Wrap an already-accepted inbound connection (from UDPListener).
    public init(acceptedConnection: NWConnection, queue: DispatchQueue) {
        self.connection = acceptedConnection
        self.queue = queue
    }

    public func start() {
        connection.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:
                if let self {
                    self.linkKind = Self.classify(self.connection.currentPath)
                    // maximumDatagramSize overreports on loopback (path MTU
                    // 16384) — the kernel still rejects sends above
                    // net.inet.udp.maxdgram (9216 stock) with EMSGSIZE, and
                    // UDP send errors are advisory, so an unclamped value
                    // means silently vanishing packets. Clamp to the sysctl.
                    let reported = self.connection.maximumDatagramSize
                    self.maxDatagramBytes = reported > 0
                        ? min(reported, Self.kernelUDPSendCeiling) : nil
                }
                self?.receiveLoop()
            case .failed(let error):
                self?.onClosed?(error)
            case .cancelled:
                self?.onClosed?(nil)
            default:
                break
            }
        }
        connection.start(queue: queue)
    }

    public func send(_ datagram: Data) {
        // .idempotent skips the per-send completion machinery — UDP send
        // errors are advisory anyway (reliability is NMP's job, Phase 2),
        // and the replay window drops the rare duplicate it permits.
        connection.send(content: datagram, completion: .idempotent)
    }

    public func batched(_ body: () -> Void) {
        connection.batch(body)
    }

    public func cancel() {
        connection.cancel()
    }

    /// Largest UDP datagram the kernel will actually send (EMSGSIZE above
    /// it). Stock Darwin ships 9216; honor a raised sysctl if present.
    static let kernelUDPSendCeiling: Int = {
        var value: CInt = 0
        var size = MemoryLayout<CInt>.size
        if sysctlbyname("net.inet.udp.maxdgram", &value, &size, nil, 0) == 0,
           value > 0 {
            return Int(value)
        }
        return 9216
    }()

    /// AWDL contention only exists on radio links; a loopback or wired path
    /// must never trigger traffic shaping. Anything ambiguous stays
    /// `.unknown` so shaping errs on the conservative side.
    private static func classify(_ path: NWPath?) -> NMPLinkKind {
        guard let path else { return .unknown }
        if path.usesInterfaceType(.wifi) || path.usesInterfaceType(.cellular) {
            return .radio
        }
        if path.usesInterfaceType(.loopback) || path.usesInterfaceType(.wiredEthernet) {
            return .wiredOrLoopback
        }
        return .unknown
    }

    private func receiveLoop() {
        connection.receiveMessage { [weak self] content, _, _, error in
            guard let self else { return }
            if let content, !content.isEmpty {
                self.onReceive?(content)
            }
            if error == nil {
                self.receiveLoop()
            } else {
                self.onClosed?(error)
            }
        }
    }
}

// MARK: - Network.framework UDP listener

/// Responder-side listener. Network.framework demultiplexes inbound UDP flows
/// by remote endpoint and surfaces each as an NWConnection, which we wrap in a
/// UDPTransport and hand to the acceptance callback (one PeerConnection each).
public final class UDPListener {

    public var onNewTransport: ((UDPTransport, NWEndpoint) -> Void)?
    public var onStateChange: ((NWListener.State) -> Void)?

    private let listener: NWListener
    private let queue: DispatchQueue

    public init(port: NWEndpoint.Port, queue: DispatchQueue) throws {
        let params = NWParameters.udp
        params.allowLocalEndpointReuse = true
        self.listener = try NWListener(using: params, on: port)
        self.queue = queue
    }

    public var port: NWEndpoint.Port? { listener.port }

    public func start() {
        listener.stateUpdateHandler = { [weak self] state in
            self?.onStateChange?(state)
        }
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            let transport = UDPTransport(acceptedConnection: connection, queue: self.queue)
            self.onNewTransport?(transport, connection.endpoint)
        }
        listener.start(queue: queue)
    }

    public func cancel() {
        listener.cancel()
    }
}
