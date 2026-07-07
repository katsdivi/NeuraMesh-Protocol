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

/// A bidirectional, unreliable, unordered datagram channel to ONE remote peer.
public protocol NMPTransport: AnyObject {
    /// Delivery callback. Invoked on the transport's queue with one complete
    /// datagram per call. Set before calling `start()`.
    var onReceive: ((Data) -> Void)? { get set }
    /// Invoked when the underlying channel dies (interface loss, etc.).
    var onClosed: ((Error?) -> Void)? { get set }

    func start()
    func send(_ datagram: Data)
    func cancel()
}

// MARK: - Network.framework UDP client transport

/// Outbound UDP transport (initiator side, or responder replying to a known
/// endpoint). One NWConnection per remote peer.
public final class UDPTransport: NMPTransport {

    public var onReceive: ((Data) -> Void)?
    public var onClosed: ((Error?) -> Void)?

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
        connection.send(content: datagram, completion: .contentProcessed { _ in
            // UDP: send errors are advisory; reliability is NMP's job (Phase 2).
        })
    }

    public func cancel() {
        connection.cancel()
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
