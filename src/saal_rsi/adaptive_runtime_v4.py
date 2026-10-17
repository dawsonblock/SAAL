"""Exact, measurable runtime requirements for freshly admitted v4 jobs."""

RUNTIME_SCHEMA = 'saal-colab-runtime-requirements-v1'
GIB = 1024 ** 3


def runtime_requirements(accelerator='A100', machine_shape='HIGH_RAM'):
    """Create the only currently admitted Colab hardware profiles."""
    if accelerator not in ('T4', 'A100'):
        raise ValueError('unsupported v4 accelerator')
    if machine_shape not in ('STANDARD', 'HIGH_RAM'):
        raise ValueError('unsupported v4 machine shape')
    gpu_minimum = {'T4': 12 * GIB, 'A100': 30 * GIB}[accelerator]
    host_minimum = 32 * GIB if machine_shape == 'HIGH_RAM' else 8 * GIB
    return {
        'schema': RUNTIME_SCHEMA,
        'accelerator': accelerator,
        'machine_shape': machine_shape,
        'minimum_gpu_memory_bytes': gpu_minimum,
        'minimum_host_memory_bytes': host_minimum,
    }


def validate_runtime_requirements(value):
    fields = {'schema', 'accelerator', 'machine_shape',
              'minimum_gpu_memory_bytes', 'minimum_host_memory_bytes'}
    if type(value) is not dict or set(value) != fields:
        raise ValueError('closed v4 runtime requirements required')
    expected = runtime_requirements(value.get('accelerator'), value.get('machine_shape'))
    if value != expected:
        raise ValueError('runtime accelerator, RAM shape or memory minimum substitution')
    return value


def colab_create_command(session, requirements):
    """Return the exact CLI request bound by a newly admitted v4 protocol."""
    validate_runtime_requirements(requirements)
    if type(session) is not str or not session.startswith('saal-'):
        raise ValueError('unique admitted Colab session name required')
    command = ['colab', 'new', '-s', session, '--gpu', requirements['accelerator']]
    if requirements['machine_shape'] == 'HIGH_RAM':
        command.append('--high-mem')
    return command


def host_memory_bytes():
    """Read total host RAM from the kernel; no third-party probe dependency."""
    with open('/proc/meminfo', 'r', encoding='ascii') as stream:
        for line in stream:
            if line.startswith('MemTotal:'):
                fields = line.split()
                if len(fields) != 3 or fields[2] != 'kB':
                    break
                return int(fields[1]) * 1024
    raise ValueError('cannot measure total host RAM')


def validate_runtime_observation(requirements, observation):
    validate_runtime_requirements(requirements)
    expected_fields = {'device', 'gpu', 'accelerator', 'gpu_memory_bytes',
                       'free_gpu_bytes_preflight', 'host_memory_bytes',
                       'high_memory_observed', 'dependency_hash_lock_enforced'}
    if type(observation) is not dict or set(observation) != expected_fields:
        raise ValueError('complete v4 runtime observation required')
    if (observation['device'] != 'cuda:0'
            or type(observation['gpu']) is not str
            or requirements['accelerator'].lower() not in observation['gpu'].lower()
            or observation['accelerator'] != requirements['accelerator']
            or type(observation['gpu_memory_bytes']) is not int
            or observation['gpu_memory_bytes'] < requirements['minimum_gpu_memory_bytes']
            or type(observation['free_gpu_bytes_preflight']) is not int
            or observation['free_gpu_bytes_preflight'] < 6 * GIB
            or type(observation['host_memory_bytes']) is not int
            or observation['host_memory_bytes'] < requirements['minimum_host_memory_bytes']
            or type(observation['high_memory_observed']) is not bool
            or (requirements['machine_shape'] == 'HIGH_RAM'
                and observation['high_memory_observed'] is not True)
            or observation['dependency_hash_lock_enforced'] is not True):
        raise ValueError('observed Colab hardware does not satisfy signed v4 requirements')
    return observation
