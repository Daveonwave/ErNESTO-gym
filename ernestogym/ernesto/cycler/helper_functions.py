from pymodbus.payload import BinaryPayloadDecoder, BinaryPayloadBuilder
from pymodbus.constants import Endian

def floats_to_modbus(values):
    """
    Converte una lista di float in un formato compatibile con Modbus.
    """
    builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
    for value in values:
        builder.add_32bit_float(value)
    return builder.to_registers()

def u16_to_modbus(values):
    """
    Converte una lista di float in un formato compatibile con Modbus.
    """
    builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
    if isinstance(values, int) :
        builder.add_16bit_uint(values)
    else:
        for value in values:
            builder.add_16bit_uint(value)
    return builder.to_registers()

def u32_to_modbus(values):
    """
    Converte una lista di float in un formato compatibile con Modbus.
    """
    builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
    for value in values:
        builder.add_32bit_uint(value)
    return builder.to_registers()

def i32_to_modbus(values):
    """
    Converte una lista di float in un formato compatibile con Modbus.
    """
    builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
    for value in values:
        builder.add_32bit_int(value)
    return builder.to_registers()
