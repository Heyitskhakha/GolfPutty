import time

from pyModbusTCP.client import ModbusClient
import struct
import numpy as np

# ---------- CONFIG ----------
DOOSAN_IP = "192.168.137.100"
DOOSAN_PORT = 502   # default Modbus TCP port
REG_START = 128    # starting register on the robot


# ---------- HELPER: float -> 2 x 16-bit registers ----------
def float_to_registers(value):
    # pack float as 4 bytes (big endian) then split into two 16-bit registers
    b = struct.pack('>f', value)
    r1 = int.from_bytes(b[0:2], byteorder='big')
    r2 = int.from_bytes(b[2:4], byteorder='big')
    return [r1, r2]

# ---------- Convert full pose ----------
regs = []
x=-36
y=-50
z=10
Ry=-10
Rz=10
Rx=-150


object_pose = np.array([x, y, z, Rx, Ry, Rz], dtype=np.float32)  # 6-element pose

# ---------- Connect and send ----------
client = ModbusClient(host=DOOSAN_IP, port=DOOSAN_PORT, auto_open=True)

if client.open():
    # write multiple registers
    # Convert float -> uint16


    for i, val in enumerate(object_pose):
        b = struct.pack('>f', val)  # float -> 4 bytes
        hi, lo = struct.unpack('>HH', b)  # 2x16-bit registers
        success = client.write_multiple_registers(128 + i * 2, [hi, lo])
        if not success:
            print(f"Failed to write registers {128 + i * 2} and {128 + i * 2 + 1}")
    if success:
        print("Pose sent successfully!")
    else:
        print("Failed to write registers.")
else:
    print("Unable to connect to Doosan Modbus TCP Slave.")
