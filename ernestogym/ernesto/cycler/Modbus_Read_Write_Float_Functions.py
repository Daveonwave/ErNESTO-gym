from pymodbus.client import ModbusTcpClient
# import pymodbus
from pymodbus.payload import BinaryPayloadDecoder, BinaryPayloadBuilder
from pymodbus.constants import Endian

# Input register Float
def read_modbus_Input_registers_Float(ip, port, start_register, num_registers):
    """
    Legge i registri Modbus TCP e restituisce i valori float decodificati.

    :param ip: Indirizzo IP del server Modbus
    :param port: Porta TCP del server Modbus
    :param start_register: Registro iniziale da leggere
    :param num_registers: Numero di registri da leggere (ogni float usa 2 registri)
    :return: Lista di valori float decodificati o None in caso di errore
    """
    client = ModbusTcpClient(host=ip, port=port)

    if client.connect():
        #print(f"✅ Connesso al server Modbus {ip}:{port}")

        result = client.read_input_registers(start_register, num_registers * 2)

        if not result.isError():
            decoder = BinaryPayloadDecoder.fromRegisters(result.registers, byteorder=Endian.BIG, wordorder=Endian.BIG)
            float_values = [decoder.decode_32bit_float() for _ in range(num_registers)]
            client.close()
            #print("🔌 Connessione chiusa.")
            return float_values
        else:
            print(f"❌ Errore nella lettura dei registri: {result}")
    else:
        print("❌ Errore: impossibile connettersi al server Modbus.")

    client.close()
    return None

# Holding register Float
def read_modbus_Holding_registers_Float(ip, port, start_register, num_registers):
    """
    Legge i registri Modbus TCP e restituisce i valori float decodificati.

    :param ip: Indirizzo IP del server Modbus
    :param port: Porta TCP del server Modbus
    :param start_register: Registro iniziale da leggere
    :param num_registers: Numero di registri da leggere (ogni float usa 2 registri)
    :return: Lista di valori float decodificati o None in caso di errore
    """
    client = ModbusTcpClient(host=ip, port=port)

    if client.connect():
        #print(f"✅ Connesso al server Modbus {ip}:{port}")

        result = client.read_holding_registers(start_register, num_registers * 2)

        if not result.isError():
            decoder = BinaryPayloadDecoder.fromRegisters(result.registers, byteorder=Endian.BIG, wordorder=Endian.BIG)
            float_values = [decoder.decode_16bit_int() for _ in range(num_registers)]
            client.close()
            #print("🔌 Connessione chiusa.")
            return float_values
        else:
            print(f"❌ Errore nella lettura dei registri: {result}")
    else:
        print("❌ Errore: impossibile connettersi al server Modbus.")

    client.close()
    return None

# Input register interi
def read_modbus_Inp_registers_Int(ip, port, start_register, num_registers):
    client = ModbusTcpClient(host=ip, port=port)
    if client.connect():
        result = client.read_input_registers(start_register, num_registers)
        if not result.isError():
            int_values = result.registers
            client.close()
            return int_values
        else:
            print(f"❌ Errore nella lettura dei registri: {result}")
    else:
        print("❌ Errore: impossibile connettersi al server Modbus.")
    client.close()
    return None



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


def write_modbus_registers(ip, port, start_register, float_values):
    """
    Scrive una lista di valori float nei registri Modbus TCP.

    :param ip: Indirizzo IP del server Modbus
    :param port: Porta TCP del server Modbus
    :param start_register: Registro iniziale per la scrittura
    :param float_values: Lista di valori float da scrivere
    :return: True se la scrittura ha successo, False altrimenti
    """
    client = ModbusTcpClient(host=ip, port=port)

    if client.connect():
        #print(f"✅ Connesso al server Modbus {ip}:{port}")

        payload = floats_to_modbus(float_values)

        result = client.write_registers(start_register, payload)

        if not result.isError():
            print(f"✅ Float scritti con successo nei registri a partire da {start_register}: {float_values}")
            # client.close()
            #print("🔌 Connessione chiusa.")
            return True
        else:
            print(f"❌ Errore durante la scrittura: {result}")
    else:
        print("❌ Errore: impossibile connettersi al server Modbus.")

    # client.close()
    return 

def write_modbus_single_register(ip, port, register, val, flag):
    """
    Scrive una lista di valori float nei registri Modbus TCP.

    :param ip: Indirizzo IP del server Modbus
    :param port: Porta TCP del server Modbus
    :param register: Registro per la scrittura
    :param float_values: Lista di valori float da scrivere
    :return: True se la scrittura ha successo, False altrimenti
    """
    client = ModbusTcpClient(host=ip, port=port)

    if client.connect():
        #print(f"✅ Connesso al server Modbus {ip}:{port}")

        if flag == 0:
            payload = floats_to_modbus(val)
        elif flag == 1:
            payload = u16_to_modbus(val)
        elif flag == 2:
            payload = u32_to_modbus(val)


        result = client.write_register(register, payload[0])

        if not result.isError():
            print(f"✅ Float scritti con successo nei registri a partire da {register}: {val}")
            # client.close()
            #print("🔌 Connessione chiusa.")
            return True
        else:
            print(f"❌ Errore durante la scrittura: {result}")
    else:
        print("❌ Errore: impossibile connettersi al server Modbus.")

    # client.close()
    return False


def write_modbus_registers_custom(ip, port, start_register, val, flag):
    """
    Scrive una lista di valori float nei registri Modbus TCP.

    :param ip: Indirizzo IP del server Modbus
    :param port: Porta TCP del server Modbus
    :param start_register: Registro iniziale per la scrittura
    :param values: Lista di valori da scrivere
    :param flag: Codifica utilizzata: 0 = floats  1 = unsigned 16-bit 2 = unsigned 32-bit
    :return: True se la scrittura ha successo, False altrimenti
    """
    client = ModbusTcpClient(host=ip, port=port)

    if client.connect():
        #print(f"✅ Connesso al server Modbus {ip}:{port}")

        if flag == 0:
            payload = floats_to_modbus(val)
        elif flag == 1:
            payload = u16_to_modbus(val)
        elif flag == 2:
            payload = u32_to_modbus(val)
        elif flag == 3:
            payload = i32_to_modbus(val)

        result = client.write_registers(start_register, payload)

        if not result.isError():
            print(f"✅ Float scritti con successo nei registri a partire da {start_register}: {val}")
            client.close()
            #print("🔌 Connessione chiusa.")
            return True
        else:
            print(f"❌ Errore durante la scrittura: {result}")
    else:
        print("❌ Errore: impossibile connettersi al server Modbus.")

    # client.close()
    return False



def write_modbus_holding_registers(ip, port, start_register, float_values):
    client = ModbusTcpClient(host=ip, port=port)

    if client.connect():
        #print(f"✅ Connesso al server Modbus {ip}:{port}")

        payload = floats_to_modbus(float_values)

        result = client.write_registers(start_register, payload)

        if not result.isError():
            print(f"✅ Float scritti con successo nei registri a partire da {start_register}: {float_values}")
            client.close()
            #print("🔌 Connessione chiusa.")
            return True
        else:
            print(f"❌ Errore durante la scrittura: {result}")
    else:
        print("❌ Errore: impossibile connettersi al server Modbus.")

    client.close()
    return False


def write_modbus_holding_registers_custom(ip, port, start_register, val, flag):
    """
    Parameters
    ----------
    ip : str
        server IP address
    port : int
        server port
    start_register : int
        Starting register address
    val : list
        Values to write
    flag : int
        Data type selector:
            0 = floats 
            1 = unsigned 16-bit
            2 = unsigned 32-bit
    """
    client = ModbusTcpClient(host=ip, port=port)

    if client.connect():
        #print(f"✅ Connesso al server Modbus {ip}:{port}")

        if flag == 0:                                       # floats 
            payload = floats_to_modbus(val)
        elif flag == 1:
            payload = u16_to_modbus(val)
        elif flag == 2:
            payload = u32_to_modbus(val)
        elif flag == 3:
            payload = i32_to_modbus(val)

        result = client.write_registers(start_register, payload)

        if not result.isError():
            print(f"✅ Float scritti con successo nei registri a partire da {start_register}: {val}")
            client.close()
            #print("🔌 Connessione chiusa.")
            return True
        else:
            print(f"❌ Errore durante la scrittura: {result}")
    else:
        print("❌ Errore: impossibile connettersi al server Modbus.")

    client.close()
    return False


# Esempio di utilizzo
#if __name__ == "__main__":
    #ip_address = '172.25.101.4'
    #port_number = 502
    #start_reg = 150
    #num_regs = 4

    #values = read_modbus_registers(ip_address, port_number, start_reg, num_regs)
    #if values:
    #    print(f"🔢 Valori float letti: {[f'{x:.2f}' for x in values]}")

    #new_values = [-34.5, 12.3, 45.7, -9.8]
    #write_modbus_registers(ip_address, port_number, start_reg, new_values)
