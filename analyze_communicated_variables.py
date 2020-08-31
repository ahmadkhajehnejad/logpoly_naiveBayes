
ROOT_FOLDER = './'

sent = 0
received = 0

with open(ROOT_FOLDER + 'center.out', 'r') as f:
    for line in f:
        s = line.split()
        if s[0] == '@@' and s[1] == 'sending' and s[3] == 'variables.':
            sent += int(s[2])

with open(ROOT_FOLDER + 'clients.out', 'r') as f:
    for line in f:
        s = line.split()
        if s[0] == '@@' and s[1] == 'sending' and s[3] == 'variables.':
            received += int(s[2])

print('      total sent variables to clients: ', sent)
print('total received variables from clients: ', received)
print('                                  sum: ', sent + received)