source activate conpy
PASSWORD=$1
echo $PASSWORD | sudo -S tcpdump -i lo -n > tcpdump.out &
python compute_communication_main.py > comp_comm.out
sleep 3
echo $PASSWORD | sudo -S killall tcpdump