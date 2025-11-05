from . import Modbus_Read_Write_Float_Functions as Mb
from pymodbus.client import ModbusTcpClient
# from pymodbus.payload import BinaryPayloadDecoder, BinaryPayloadBuilder
from pymodbus.constants import Endian
from . import helper_functions as hfunc
import numpy as np
import warnings
import time
import threading
from apscheduler.schedulers.background import BackgroundScheduler
import datetime



class Cycler:
    '''
    Main function to control the cycler
    '''

    def __init__(self, ip = '192.168.1.191', port=502):
        self._lock = threading.Lock()
        self._scheduler = BackgroundScheduler()
        self._job = None

        # Control parameters
        self.ip = ip
        self.port = port
        self.client = None
        self.connected = False
        self._P_set = None
        self._I_max = None
        self._V_min = None
        self._V_max = None
        self._update_interval = 1.0
        self.V_read = None
        self.I_read = None

        '''Set CC mode'''
        self.set_mode(1)

        '''Set I setpoint to 0A'''
        self.set_I_setpoint(0)

        # self._stop_event = threading.Event()
        # self._thread = None
        # self._lock = threading.Lock()  # To  update P_set


    # ----------------------------
    # Connection Management
    # ----------------------------
    def connect(self):
        self.client = ModbusTcpClient(host=self.ip, port=self.port)
        self.connected = self.client.connect()
        if not self.connected:
            raise ConnectionError(f"Could not connect to {self.ip}:{self.port}")
        print(f"[INFO] Connected to cycler at {self.ip}:{self.port}")

    def disconnect(self):
        if self.client:
            self.client.close()
            self.connected = False
            print("[INFO] Connection closed.")

    # ----------------------------
    # Wrapper for existing function
    # ----------------------------

    def write_modbus_multiple_registers(self, start_register, new_values, flag):
        """
        Scrive una lista di valori float nei registri Modbus TCP.
        :param start_register: Registro iniziale per la scrittura
        :param new_values: Lista di valori da scrivere
        :param flag: Codifica utilizzata: 0 = floats  1 = unsigned 16-bit 2 = unsigned 32-bit
        :return: True se la scrittura ha successo, False altrimenti
        """
        if not self.connected:
            self.connect()  # auto-connect


        #print(f"✅ Connesso al server Modbus {ip}:{port}")

        if flag == 0:
            payload = hfunc.floats_to_modbus(new_values)
        elif flag == 1:
            payload = hfunc.u16_to_modbus(new_values)
        elif flag == 2:
            payload = hfunc.u32_to_modbus(new_values)
        elif flag == 3:
            payload = hfunc.i32_to_modbus(new_values)
        else:
            raise ValueError(f"Unsupported flag: {flag}")


        result = self.client.write_registers(start_register, payload)
        
        if not result.isError():
            print(f"✅ Float scritti con successo nei registri a partire da {start_register}: {new_values}")
            
            # self.disconnect()
            #print("🔌 Connessione chiusa.")
            return True
        else:
            print(f"❌ Errore durante la scrittura: {result}")

        # client.close()
        return False
    

        # Holding register Float

    def read_modbus_Holding_registers_Float(self, start_register, num_registers, flag):
        """
        Legge i registri Modbus TCP e restituisce i valori float decodificati.

        :param ip: Indirizzo IP del server Modbus
        :param port: Porta TCP del server Modbus
        :param start_register: Registro iniziale da leggere
        :param num_registers: Numero di registri da leggere (ogni float usa 2 registri)
        :return: Lista di valori float decodificati o None in caso di errore
        """
        if not self.connected:
            self.connect()  # auto-connect


        # result = self.client.read_holding_registers(start_register, num_registers * 2)
        result = self.client.read_holding_registers(start_register, num_registers * 2)

        if not result.isError():
            
            # decoder = BinaryPayloadDecoder.fromRegisters(result.registers, byteorder=Endian.BIG, wordorder=Endian.BIG)
            # float_values = [decoder.decode_16bit_int() for _ in range(num_registers)]



            float_values = ModbusTcpClient.convert_from_registers(registers = result.registers, data_type=ModbusTcpClient.DATATYPE.INT16, word_order = "big")

            # if flag == 0:
                #     float_values = [decoder.decode_32bit_float() for _ in range(num_registers)] 
                # elif flag == 1:
                #     float_values = [decoder.decode_16bit_uint() for _ in range(num_registers)]
                # elif flag == 2:
                #     float_values = [decoder.decode_32bit_uint() for _ in range(num_registers)] 
                # elif flag == 3:
                #     float_values = [decoder.decode_32bit_int() for _ in range(num_registers)] 

                # client.close()
                #print("🔌 Connessione chiusa.")
            return float_values
        # else:
        #         print(f"❌ Errore nella lettura dei registri: {result}")
        else:
            print("❌ Errore: impossibile connettersi al server Modbus.")

        # client.close()
        return None
    
    def read_modbus_Writing_registers_Float(self, start_register, num_registers, flag):
        """
        Legge i registri Modbus TCP e restituisce i valori float decodificati.

        :param ip: Indirizzo IP del server Modbus
        :param port: Porta TCP del server Modbus
        :param start_register: Registro iniziale da leggere
        :param num_registers: Numero di registri da leggere (ogni float usa 2 registri)
        :return: Lista di valori float decodificati o None in caso di errore
        """
        if not self.connected:
            self.connect()  # auto-connect

        # determine number of registers to read
        count = num_registers * 2 if flag in [0,2,3] else num_registers

        result = self.client.read_holding_registers(address=start_register, count = count)
        # result = self.client.read_holding_registers(start_register, num_registers * 2)
        # result = self.client.read_holding_registers(start_register)

        if not result.isError():
            # decoder = BinaryPayloadDecoder.fromRegisters(result.registers, byteorder=Endian.BIG, wordorder=Endian.BIG)
            # float_values = [decoder.decode_16bit_int() for _ in range(num_registers)]

            if flag == 0:
                # float_values = [decoder.decode_32bit_float() for _ in range(num_registers)] 
                values = ModbusTcpClient.convert_from_registers(registers = result.registers, data_type=ModbusTcpClient.DATATYPE.FLOAT32)
            elif flag == 1:
                # float_values = [decoder.decode_16bit_uint() for _ in range(num_registers)]
                values = ModbusTcpClient.convert_from_registers(registers = result.registers, data_type=ModbusTcpClient.DATATYPE.UINT16)
            elif flag == 2:
                # float_values = [decoder.decode_32bit_uint() for _ in range(num_registers)] 
                values = ModbusTcpClient.convert_from_registers(registers = result.registers, data_type=ModbusTcpClient.DATATYPE.UINT32)
            elif flag == 3:
                # float_values = [decoder.decode_32bit_int() for _ in range(num_registers)] 
                values = ModbusTcpClient.convert_from_registers(registers = result.registers, data_type=ModbusTcpClient.DATATYPE.INT32)

                # client.close()
                #print("🔌 Connessione chiusa.")
            return values
        # else:
        #         print(f"❌ Errore nella lettura dei registri: {result}")
        else:
            print("❌ Errore: impossibile connettersi al server Modbus.")

        # client.close()
        return None


    # ----------------------------
    # High-level Commands
    # ----------------------------
    def start_operation(self):
        self.write_modbus_multiple_registers(start_register=2001, new_values=[1], flag=1)

    def stop_operation(self):
        self.write_modbus_multiple_registers(start_register=2001, new_values=[0], flag=1)

    def exit_communications(self):
        self.write_modbus_multiple_registers(start_register=2000, new_values=[0], flag=1)
    
    def clear_faults(self):
        self.write_modbus_multiple_registers(start_register = 2002, new_values=[1], flag = 1)

    def set_mode(self, mode=1):
        '''Set the mode function:
            0: SOURCE_CV mode
            1: SOURCE_CC mode
            2: Battery test mode (BT_CC)
            3: Battery simulation mode (BS_CV)
            5: IV mode
            6: Time scaling
            7: Dynamic MPPT mode
            8: Shading
            20: LOAD-CC mode
            21: LOAD-CV mode
            22: LOAD-CP mode
            23: LOAD-CR mode
            24: LOAD-CVCC mode
            25: LOAD-CVCR mode
            26: LOAD-CRCC mode
            27: LOAD-AUTO mode
            100: List
            101: UDW
            102. Rectangle
            103: Triangle
            104: Sine wave
            200-219: Vehicle electronics waveform
        '''
        self.write_modbus_multiple_registers(start_register = 2050, new_values=[mode], flag = 1)

    def set_I_setpoint(self,current):
        # Check type
        if not isinstance(current, (int, float)):
            raise TypeError("Current setpoint must be a number (in amperes).")

        # Prevent double conversion
        if abs(current) > 1000:
            raise ValueError(
                f"Invalid current setpoint: {current}. "
                "Value likely already converted to integer units (should be in amperes)."
            )

        # Convert and write to Modbus
        current_set = int(current * 10000)
        self.write_modbus_multiple_registers(start_register = 2330, new_values=[current_set], flag = 3)

    def battery_test_mode_chdch(self, mode):
        self.write_modbus_multiple_registers(start_register = 2360, new_values=[mode], flag = 1)

    def battery_test_mode_power(self, P):
        ch_mode = self.read_modbus_Writing_registers_Float(start_register=2360,num_registers=1, flag = 1)
        if ch_mode ==0:
            self.write_modbus_multiple_registers(start_register = 2373, new_values=[P], flag = 2)
        elif ch_mode ==1:
            self.write_modbus_multiple_registers(start_register = 2400, new_values=[P], flag = 2)
        else:
            raise ValueError('Charge mode not recognized!')

    def battery_test_mode_current(self, I):
        self.write_modbus_multiple_registers(start_register = 2369, new_values=[I], flag = 2)

    def dc_load_P_setpoint(self,P):
        self.write_modbus_multiple_registers(start_register = 4014, new_values=[P], flag = 2)


    def read_mode(self):
        return self.read_modbus_Writing_registers_Float(start_register=2050,num_registers=1, flag = 1)

    def read_P_meas(self):
        return self.read_modbus_Writing_registers_Float(start_register=1104,num_registers=1, flag = 3)/10

    def read_V_meas(self):
        return self.read_modbus_Writing_registers_Float(start_register=1100,num_registers=1, flag = 3)/10000

    def read_I_meas(self):
        return self.read_modbus_Writing_registers_Float(start_register=1102,num_registers=1, flag = 3)/10000

    def triple_reading(self):
        read = [self.read_V_meas(), self.read_I_meas(), self.read_P_meas()]
        return read
    
    def set_I_from_P(self, P_set, I_max, V_min, V_max):
        V_read = self.read_V_meas()
        if V_read < V_min or V_read > V_max:
            self.stop_operation()
            raise ValueError('Error in V reading, operation stopped')
        else:
            I_computed = P_set / V_read
            I_set = np.clip(I_computed,-I_max, I_max,)
            if I_computed < -I_max or I_computed > I_max:
                warnings.warn(f"I_computed = {I_computed:.3f} A was clipped to I_set = {I_set:.3f} A")
        self.set_I_setpoint(I_set)
    
    def reading_for_N(self, N):
        n = 0
        while n < N:
            print(self.triple_reading())
            time.sleep(0.3)
            n += 1

    def _follow_P_loop(self):
        # This is the background control loop that continuously updates
        # the current (I_set) to maintain the desired power (P_set).
        """Single iteration of the power control loop."""
        with self._lock:
            # Stop if no P_set defined
            if self._P_set is None:
                return
            
            P_set = self._P_set
            I_max = self._I_max
            V_min = self._V_min
            V_max = self._V_max

        try:
            self.V_read = self.read_V_meas()
            # print(f"\nVoltage = ({self.V_read:.3f} V)\n")

            # Safety check
            if self.V_read < V_min or self.V_read > V_max:
                self.stop_follow_P()
                warnings.warn(f"Voltage out of bounds ({self.V_read:.3f} V).")
                return

            if abs(self.V_read) < 1e-6:
                warnings.warn("Voltage too low; skipping update.")
                return

            I_computed = P_set / self.V_read
            I_set = np.clip(I_computed, -I_max, I_max)

            if I_computed != I_set:
                warnings.warn(f"I_computed={I_computed:.3f} A clipped to {I_set:.3f} A")

            self.set_I_setpoint(I_set)
            # self.start_operation()
            self.I_read = I_set

            
            print(f"Current = ({self.I_read:.3f} A)\n")
            print(f"Power = ({P_set:.3f} W)\n\n")

            

        except Exception as e:
            self.stop_follow_P()
            warnings.warn(f"Error in control loop: {e}")


    # Public method to start following a new P_set
    def start_follow_P(self, P_set, I_max, V_min, V_max, duration=None, update_interval = 1):
        """
        Configure and start the background loop that adjusts current
        to maintain the desired power (P_set).
        
        Parameters:
            P_set : float
                Target power in watts.
            I_max : float
                Maximum allowable current (A), used for clipping and protection.
            V_min, V_max : float
                Allowed voltage range (V) for safe operation.
            duration : float, optional
                Duration of the imposed setpoint (seconds).
                Default is inf.
            update_interval : float, optional
                Time interval between control updates (seconds).
                Default is 1.0 s.
        """

        """Set the setpoint and the other params"""
        with self._lock:
            self._P_set = P_set
            self._I_max = I_max
            self._V_min = V_min
            self._V_max = V_max
            self._update_interval = update_interval

        '''If cycler control loop already running'''
        if self._job is not None:
            self._scheduler.reschedule_job(
                self._job.id,
                trigger="interval",
                seconds=self._update_interval,
                end_date=(datetime.datetime.now() + datetime.timedelta(seconds=duration)) if duration is not None else None
            )
            print(f"[Scheduler] Power control job rescheduled (power={P_set} W).")
            return
        
        '''Frist instance'''
        self._scheduler.start()
        self._job = self._scheduler.add_job(
            self._follow_P_loop,
            "interval",
            seconds=self._update_interval,
            max_instances=1,
            coalesce=True,
            end_date=(datetime.datetime.now() + datetime.timedelta(seconds=duration)) if duration is not None else None
        )

        self.start_operation()

        print(f"[Scheduler] Power control started for duration={duration} s.")


    # Stop following P_set
    def stop_follow_P(self):
        """Stop the power control loop."""
        if self._job:
            self._job.remove()
            self._job = None

        if self._scheduler.running:
            self._scheduler.remove_all_jobs()
            self._scheduler.shutdown(wait=False)

        print("[Scheduler] Power control stopped.")
        self.stop_operation()
        print("[Cycler] Power control stopped.")

        # Recreate scheduler for next start
        self._scheduler = BackgroundScheduler()

    
            
# my_cycler.set_I_from_P(4, 5, 3.2, 4.14)
    # def set_voltage(self, voltage):
    #     VOLT_REG = 100  # example
    #     return self.write_register(VOLT_REG, voltage, flag=0)

    # def set_current(self, current):
    #     CURR_REG = 101  # example
    #     return self.write_register(CURR_REG, current, flag=0)

    # def start_charge(self):
    #     CMD_REG = 200
    #     return self.write_register(CMD_REG, 1, flag=1)  # 1 = charge

    # def stop(self):
    #     CMD_REG = 200
    #     return self.write_register(CMD_REG, 0, flag=1)  # 0 = stop
