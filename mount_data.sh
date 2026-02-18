sshfs -o reconnect \
      -o ServerAliveInterval=15 \
      -o ServerAliveCountMax=3 \
      -o allow_other \
      umang@[vrl-h200.ece.ucsb.edu]:/data/umang/materials /data/home/umang/Materials/Materials_data_mount
      