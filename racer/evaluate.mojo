"""Deterministic evaluation, with statistics transferred once at the end."""
from .core import Device, Params, Ptr
from .env import OBS, BEAMS, reward_offset, step, observe
from .network import memory, forward
from .ppo import input, sample


def accumulate(i: Int, data: Ptr, map: Ptr, p: Params):
    var n = Int(p.envs)
    var mem = memory(n)
    var result = reward_offset(n) + i * 2
    var stats = mem.obs + i * 3
    data[unsafe_offset=stats] += data[unsafe_offset=result]
    if data[unsafe_offset=result + 1] == 1:
        data[unsafe_offset=stats + 1] += 1
    elif data[unsafe_offset=result + 1] == 3:
        data[unsafe_offset=stats + 2] += 1


def evaluate(mut device: Device, n: Int, steps: Int, seed: UInt32) raises:
    for t in range(steps):
        device.run[input](n * OBS, Params(Int32(n), seed, -1))
        forward(device, n, n, False)
        device.run[sample](n, Params(Int32(n), seed, 0, Int32(t), 1))
        device.run[step](n, Params(Int32(n), seed))
        device.run[accumulate](n, Params(Int32(n), seed))
        device.run[observe](n * BEAMS, Params(Int32(n), seed))
    var stats = device.data.create_sub_buffer[DType.float32](memory(n).obs, n * 3)
    var reward: Float64 = 0
    var collisions: Float64 = 0
    var finishes: Float64 = 0
    with stats.map_to_host() as host:
        for i in range(n):
            reward += Float64(host[i * 3])
            collisions += Float64(host[i * 3 + 1])
            finishes += Float64(host[i * 3 + 2])
    print(
        "reward/step",
        reward / Float64(n * steps),
        "failures/car",
        collisions / Float64(n),
        "finishes/car",
        finishes / Float64(n),
    )
