import config.general

def load_valid_src_trg_pairts():

    valid_src_trg_pairs = set()

    with open('comp_comm.out', 'r') as file:
        lines = file.readlines()

    for s in lines:

        if not s.startswith('## from_port_to_port:'):
            continue

        parts = s.split()
        src_port = int(parts[3][:-1])
        trg_port = int(parts[5])
        valid_src_trg_pairs.update([(src_port, trg_port)])

    return valid_src_trg_pairs



if __name__ == '__main__':

    valid_src_trg_pairs = load_valid_src_trg_pairts()

    total_sent_bytes = 0
    total_received_bytes = 0

    with open('tcpdump.out','r') as file:
        lines = file.readlines()

    for s in lines:

        parts = s.split()
        if len(parts) < 8:
            continue

        if parts[-2] != 'length':
            continue

        src = parts[2]
        if not src.startswith('127.0.0.1.'):
            continue
        src_port = int(src[10:])

        trg = parts[4]
        if not trg.startswith('127.0.0.1.'):
            continue
        trg_port = int(trg[10:-1])

        if (src_port, trg_port) not in valid_src_trg_pairs:
            continue

        if config.general.first_client_port <= trg_port < config.general.first_client_port + config.general.num_clients:
            total_sent_bytes +=  int(parts[-1])
        else:
            total_received_bytes += int(parts[-1])

    print('total sent bytes: ', total_sent_bytes)
    print('total rcvd bytes: ', total_received_bytes)
    print('sum sent & recvd: ', total_sent_bytes + total_received_bytes)
