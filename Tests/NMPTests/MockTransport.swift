//
//  MockTransport.swift
//  NMPTests
//
//  Deterministic in-memory datagram transport for integration tests.
//  Supports drop and corruption hooks to simulate lossy links.
//

import Foundation
@testable import NMP

final class MockTransport: NMPTransport {

    var onReceive: ((Data) -> Void)?
    var onClosed: ((Error?) -> Void)?

    /// Return true to drop the outbound datagram (loss simulation).
    var dropOutbound: ((Data) -> Bool)?
    /// Optionally mutate outbound datagrams (corruption simulation).
    var corruptOutbound: ((Data) -> Data)?

    weak var peer: MockTransport?
    private let deliveryQueue: DispatchQueue
    private let lock = NSLock()
    private var _sentDatagrams: [Data] = []

    var sentDatagrams: [Data] {
        lock.lock(); defer { lock.unlock() }
        return _sentDatagrams
    }

    init(deliveryQueue: DispatchQueue) {
        self.deliveryQueue = deliveryQueue
    }

    static func pair(label: String = "mock") -> (MockTransport, MockTransport) {
        let qA = DispatchQueue(label: "\(label).a")
        let qB = DispatchQueue(label: "\(label).b")
        let a = MockTransport(deliveryQueue: qA)
        let b = MockTransport(deliveryQueue: qB)
        a.peer = b
        b.peer = a
        return (a, b)
    }

    func start() {}
    func cancel() {}

    func send(_ datagram: Data) {
        lock.lock()
        _sentDatagrams.append(datagram)
        lock.unlock()

        if dropOutbound?(datagram) == true { return }
        let wire = corruptOutbound?(datagram) ?? datagram
        guard let peer else { return }
        peer.deliveryQueue.async {
            peer.onReceive?(wire)
        }
    }

    /// Inject an arbitrary datagram into this transport's receive path,
    /// as if it arrived off the wire (malformed-input / duplicate tests).
    func inject(_ datagram: Data) {
        deliveryQueue.async { [weak self] in
            self?.onReceive?(datagram)
        }
    }
}
