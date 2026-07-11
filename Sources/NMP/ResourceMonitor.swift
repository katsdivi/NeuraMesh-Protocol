//
//  ResourceMonitor.swift
//  NMP — Mesh 2.1
//
//  Live host resource sampling for the Devices panel: RAM, storage, CPU,
//  and this process's own footprint — all read from Mach/BSD interfaces
//  (host_statistics64, task_info, statfs, sysctl), no shelling out, no
//  third-party code.
//
//  MEASUREMENT HONESTY: every number here is a real kernel counter for
//  THIS machine. The dashboard's mesh peers are in-process (Phase 8's
//  in-memory links), so all peers genuinely share this host — the API
//  says so explicitly instead of inventing per-peer "device" stats. When
//  a physical peer joins a future mesh, its own monitor runs there and
//  reports over the wire.
//
//  CPU% is a delta between two samples of the host tick counters, so the
//  FIRST sample after init reports cpuPercent = nil (there is nothing to
//  diff against yet). Callers polling every couple of seconds get real
//  utilization from the second sample on.
//
//  Threading: `sample()` is synchronized internally — callable from any
//  queue.
//

import Foundation
#if canImport(IOKit)
import IOKit
#endif

/// One point-in-time reading of this host's resources.
public struct NMPHostResourceSample: Sendable {
    public let hostname: String

    // RAM (bytes) — host-wide, from host_statistics64.
    public let ramTotalBytes: UInt64
    /// active + wired + compressed: memory actually in use (the standard
    /// "used" definition; free + inactive are reclaimable).
    public let ramUsedBytes: UInt64
    /// This process's physical footprint (what Activity Monitor shows) —
    /// watching this move while a model loads is the direct way to verify
    /// the mesh is really taking the RAM it claims.
    public let processFootprintBytes: UInt64

    // Storage (bytes) — the volume holding the user's home directory.
    public let storageTotalBytes: UInt64
    public let storageFreeBytes: UInt64

    /// Host CPU utilization 0...100 across all cores; nil on the first
    /// sample (tick deltas need two readings).
    public let cpuPercent: Double?
    /// GPU utilization 0...100, whole machine, straight from the
    /// accelerator driver's own counter ("Device Utilization %" in the
    /// IOAccelerator performance statistics). nil where the counter does
    /// not exist (non-macOS, or a driver that does not publish it).
    /// NOTE: whole-GPU — there is no public per-process split; the
    /// reference engine is pure CPU, so expect this to move only under a
    /// Metal workload (llama.cpp) or other GPU use on the machine.
    public let gpuPercent: Double?
    public let sampledAt: Date

    public var ramUsedPercent: Double {
        ramTotalBytes > 0 ? Double(ramUsedBytes) / Double(ramTotalBytes) * 100 : 0
    }

    public var storageUsedPercent: Double {
        guard storageTotalBytes > 0 else { return 0 }
        return Double(storageTotalBytes - storageFreeBytes)
            / Double(storageTotalBytes) * 100
    }

    /// JSON shape served by GET /api/devices/metrics.
    public var asJSONObject: [String: Any] {
        var object: [String: Any] = [
            "hostname": hostname,
            "ram_total_mb": Int(ramTotalBytes / (1 << 20)),
            "ram_used_mb": Int(ramUsedBytes / (1 << 20)),
            "ram_used_percent": (ramUsedPercent * 10).rounded() / 10,
            "process_footprint_mb": Int(processFootprintBytes / (1 << 20)),
            "storage_total_gb": (Double(storageTotalBytes) / Double(1 << 30) * 10)
                .rounded() / 10,
            "storage_free_gb": (Double(storageFreeBytes) / Double(1 << 30) * 10)
                .rounded() / 10,
            "storage_used_percent": (storageUsedPercent * 10).rounded() / 10,
        ]
        if let cpuPercent {
            object["cpu_percent"] = (cpuPercent * 10).rounded() / 10
        }
        if let gpuPercent {
            object["gpu_percent"] = (gpuPercent * 10).rounded() / 10
        }
        return object
    }
}

/// Samples host resources; holds the previous CPU tick counters so
/// successive samples yield real utilization deltas.
public final class NMPResourceMonitor {

    private let lock = NSLock()
    private var previousTicks: (busy: UInt64, total: UInt64)?

    public init() {}

    public func sample() -> NMPHostResourceSample {
        lock.lock()
        defer { lock.unlock() }
        return NMPHostResourceSample(
            hostname: NMPLANIdentity.localHostname(),
            ramTotalBytes: Self.physicalMemoryBytes(),
            ramUsedBytes: Self.usedMemoryBytes(),
            processFootprintBytes: Self.processFootprintBytes(),
            storageTotalBytes: Self.storage().total,
            storageFreeBytes: Self.storage().free,
            cpuPercent: cpuPercentLocked(),
            gpuPercent: Self.gpuUtilizationPercent(),
            sampledAt: Date())
    }

    // MARK: RAM

    static func physicalMemoryBytes() -> UInt64 {
        ProcessInfo.processInfo.physicalMemory
    }

    /// active + wired + compressed pages, via host_statistics64.
    static func usedMemoryBytes() -> UInt64 {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.stride
                / MemoryLayout<integer_t>.stride)
        let result = withUnsafeMutablePointer(to: &stats) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        let pageSize = UInt64(vm_kernel_page_size)
        let usedPages = UInt64(stats.active_count)
            + UInt64(stats.wire_count)
            + UInt64(stats.compressor_page_count)
        return usedPages * pageSize
    }

    /// phys_footprint via task_info — Activity Monitor's "Memory" column.
    static func processFootprintBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.stride
                / MemoryLayout<integer_t>.stride)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return info.phys_footprint
    }

    // MARK: Storage

    static func storage() -> (total: UInt64, free: UInt64) {
        var fs = statfs()
        guard statfs(NSHomeDirectory(), &fs) == 0 else { return (0, 0) }
        let blockSize = UInt64(fs.f_bsize)
        // f_bavail: blocks available to non-root — what a user can use.
        return (total: UInt64(fs.f_blocks) * blockSize,
                free: UInt64(fs.f_bavail) * blockSize)
    }

    // MARK: GPU (macOS: the accelerator driver's own utilization counter)

    /// "Device Utilization %" from the IOAccelerator registry entry's
    /// PerformanceStatistics — the same counter Activity Monitor's GPU
    /// history reads. Whole-machine; max across accelerators on the odd
    /// multi-GPU Mac. nil off macOS or if the driver omits the key.
    static func gpuUtilizationPercent() -> Double? {
        #if canImport(IOKit) && os(macOS)
        var iterator = io_iterator_t()
        guard IOServiceGetMatchingServices(
            kIOMainPortDefault,
            IOServiceMatching("IOAccelerator"),
            &iterator) == KERN_SUCCESS else { return nil }
        defer { IOObjectRelease(iterator) }

        var utilization: Double?
        while true {
            let entry = IOIteratorNext(iterator)
            guard entry != 0 else { break }
            defer { IOObjectRelease(entry) }
            var propertiesRef: Unmanaged<CFMutableDictionary>?
            guard IORegistryEntryCreateCFProperties(
                entry, &propertiesRef, kCFAllocatorDefault, 0) == KERN_SUCCESS,
                let properties = propertiesRef?.takeRetainedValue() as? [String: Any],
                let statistics = properties["PerformanceStatistics"] as? [String: Any]
            else { continue }
            if let value = statistics["Device Utilization %"] as? NSNumber {
                utilization = Swift.max(utilization ?? 0,
                                        Swift.min(100, Swift.max(0, value.doubleValue)))
            }
        }
        return utilization
        #else
        return nil
        #endif
    }

    // MARK: CPU (host-wide tick deltas; caller holds `lock`)

    private func cpuPercentLocked() -> Double? {
        guard let ticks = Self.cpuTicks() else { return nil }
        defer { previousTicks = ticks }
        guard let previous = previousTicks,
              ticks.total > previous.total else { return nil }
        let busy = Double(ticks.busy - previous.busy)
        let total = Double(ticks.total - previous.total)
        return min(100, max(0, busy / total * 100))
    }

    static func cpuTicks() -> (busy: UInt64, total: UInt64)? {
        var load = host_cpu_load_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<host_cpu_load_info_data_t>.stride
                / MemoryLayout<integer_t>.stride)
        let result = withUnsafeMutablePointer(to: &load) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        let user = UInt64(load.cpu_ticks.0)      // CPU_STATE_USER
        let system = UInt64(load.cpu_ticks.1)    // CPU_STATE_SYSTEM
        let idle = UInt64(load.cpu_ticks.2)      // CPU_STATE_IDLE
        let nice = UInt64(load.cpu_ticks.3)      // CPU_STATE_NICE
        let busy = user + system + nice
        return (busy: busy, total: busy + idle)
    }
}
