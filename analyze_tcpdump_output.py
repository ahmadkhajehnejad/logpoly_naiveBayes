import config.general
from config.client_nodes_address import client_nodes_address

ROOT_FOLDER = './results_communication/kde-n10000/'

def load_valid_src_trg_pairts():

    valid_src_trg_pairs = set()

    for filename in ['clients.out', 'center.out']:
        with open(ROOT_FOLDER + filename, 'r') as file:
            lines = file.readlines()

        for s in lines:

            if not s.startswith('## from_port_to_port:'):
                continue

            parts = s.split()

            src_adress = parts[2] + '.' + parts[4]
            trg_address = parts[6] + '.' + parts[8]
            valid_src_trg_pairs.update([(src_adress, trg_address)])

    return valid_src_trg_pairs



if __name__ == '__main__':

    valid_src_trg_pairs = load_valid_src_trg_pairts()

    total_sent_bytes = 0
    total_received_bytes = 0
    packet_count = 0
    non_empty_packet_count = 0

    with open(ROOT_FOLDER + 'tcpdump.out','r') as file:
        lines = file.readlines()

    for s in lines:

        parts = s.split()
        if len(parts) < 8:
            continue

        if parts[-2] != 'length':
            continue

        src_address = parts[2]

        trg_address = parts[4][:-1]

        if (src_address, trg_address) not in valid_src_trg_pairs:
            continue

        if trg_address == client_nodes_address[0] + '.' + str(client_nodes_address[1]):
            total_sent_bytes +=  int(parts[-1])
        else:
            total_received_bytes += int(parts[-1])

        packet_count += 1
        if int(parts[-1]) > 0:
            non_empty_packet_count += 1

    print('     total sent bytes to clients: ', total_sent_bytes)
    print('total receivd bytes from clients: ', total_received_bytes)
    print('                             sum: ', total_sent_bytes + total_received_bytes)
    print('          packet count: ', packet_count)
    print('non-empty packet count: ', non_empty_packet_count)
