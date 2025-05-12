import scapy.all as scapy
import socket
import netifaces as ni
from ipaddress import ip_network
from concurrent.futures import ThreadPoolExecutor

# Passo 1: Detecta a rede atual conectada
def get_connected_network():
    try:
        default_iface = ni.gateways()['default'][ni.AF_INET][1]
        iface_info = ni.ifaddresses(default_iface)[ni.AF_INET][0]
        ip = iface_info['addr']
        netmask = iface_info['netmask']
        prefix = sum([bin(int(x)).count('1') for x in netmask.split('.')])
        network = f"{ip}/{prefix}"
        return network
    except Exception as e:
        print("Erro ao detectar a rede:", e)
        return None

# Passo 2: Scanner ARP para identificar dispositivos conectados
def scan(ip_range):
    arp_request = scapy.ARP(pdst=ip_range)
    broadcast = scapy.Ether(dst="ff:ff:ff:ff:ff:ff")
    answered = scapy.srp(broadcast / arp_request, timeout=2, verbose=False)[0]

    devices = []
    for sent, received in answered:
        ip = received.psrc
        mac = received.hwsrc
        try:
            hostname = socket.gethostbyaddr(ip)[0]
        except:
            hostname = "Desconhecido"
        devices.append({'ip': ip, 'mac': mac, 'hostname': hostname})
    return devices

# Passo 3: Scanner de portas
def scan_ports(ip, ports=[22, 80, 443, 445, 3389, 8080, 3306, 8000, 5432, 5000, 21]):
    open_ports = []
    for port in ports:
        try:
            sock = socket.create_connection((ip, port), timeout=0.5)
            open_ports.append(port)
            sock.close()
        except:
            pass
    return open_ports

# Execução principal
if __name__ == "__main__":
    ip_range = get_connected_network()
    if not ip_range:
        exit()

    print(f"\n🔍 Escaneando dispositivos na rede: {ip_range}\n")
    devices = scan(ip_range)

    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(scan_ports, d['ip']): d for d in devices}
        for future in futures:
            device = futures[future]
            ports = future.result()
            print(f"📡 IP: {device['ip']}")
            print(f"🖥️  Hostname: {device['hostname']}")
            print(f"🔗 MAC: {device['mac']}")
            print(f"🧩 Portas abertas: {ports if ports else 'Nenhuma'}")
            print("-" * 40)
