"""For tensotflow v1.x"""

def tensorflow_v1_dynamic_mem_allocation():
    from tensorflow.compat.v1 import (
        ConfigProto,
        InteractiveSession,
    )

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

"""For tensotflow v2.x"""

def tensorflow_v2_dynamic_mem_allocation(): 
    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices('GPU')

    try:
        tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)
    except (ValueError, RuntimeError) as e:
        raise e

