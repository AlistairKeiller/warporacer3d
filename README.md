# warporacer3d

Mojo simulation and PPO; Python map preparation and a WebGL viewer.

```sh
uv sync
uv run python viewer.py
uv run python viewer.py flat --device cpu --cars 1
uv run python viewer.py ramp --checkpoint build/agent.wrppo --cars 16
```

W/S: throttle · A/D: steer · Space: pause · R: reset.

```sh
uv run python prepare.py ramp build/ramp.wrmap
uv run mojo build main.mojo -o build/racer
build/racer train build/ramp.wrmap gpu 256 300 build/agent.wrppo
build/racer eval build/ramp.wrmap gpu 256 1000 build/agent.wrppo
build/racer benchmark build/ramp.wrmap gpu 1024 1000
build/racer train build/ramp.wrmap cpu 32 100 build/agent.wrppo build/agent.wrppo
```

`MODE MAP [auto|gpu|cpu] [CARS] [STEPS_OR_ITERATIONS] [CHECKPOINT] [RESUME]`

Built-in maps: `flat`, `ramp`, `bank`. Mesh and ROS map YAML also work:

```sh
uv run python prepare.py ../warporacer/maps/my_map.yaml build/my_map.wrmap
uv run mojo build main.mojo -D CPU_ONLY=true -o build/racer-cpu
uv run python -m unittest discover -s tests -v
uv run python prepare.py flat build/flat.wrmap
uv run mojo -I . tests/test_mojo.mojo build/flat.wrmap build/ramp.wrmap
uv run mojo -I . tests/gradient.mojo cpu build/flat.wrmap build/cpu-gradient.bin
uv run mojo -I . tests/gradient.mojo gpu build/flat.wrmap build/gpu-gradient.bin
uv run --with torch python tests/check_gradient.py build/cpu-gradient.bin build/gpu-gradient.bin
```
