sshfs -o reconnect \
      -o ServerAliveInterval=15 \
      -o ServerAliveCountMax=3 \
      -o allow_other \
      [warrenz]@[vrl-h200.ece.ucsb.edu]:/data/warrenz/materials /data/warren/materials/materials_data_mount
      
# (base) umang@vrl-h200:~$ sudo systemctl status sshd
# [sudo] password for umang: 
# Unit sshd.service could not be found.
# (base) umang@vrl-h200:~$ sudo ufw status
# Status: active

# To                         Action      From
# --                         ------      ----
# 22                         ALLOW       128.111.0.0/16            
# 2049                       ALLOW       128.111.56.136            
# 2049/tcp                   ALLOW       Anywhere                  
# 111/tcp                    ALLOW       Anywhere                  
# 111/udp                    ALLOW       Anywhere                  
# 20048/tcp                  ALLOW       Anywhere                  
# 875/tcp                    ALLOW       Anywhere                  
# 662/tcp                    ALLOW       Anywhere                  
# 32803/tcp                  ALLOW       Anywhere                  
# 7263                       ALLOW       Anywhere                  
# 7262                       ALLOW       Anywhere                  
# 8000/tcp                   ALLOW       Anywhere                  
# 2049/tcp (v6)              ALLOW       Anywhere (v6)             
# 111/tcp (v6)               ALLOW       Anywhere (v6)             
# 111/udp (v6)               ALLOW       Anywhere (v6)             
# 20048/tcp (v6)             ALLOW       Anywhere (v6)             
# 875/tcp (v6)               ALLOW       Anywhere (v6)             
# 662/tcp (v6)               ALLOW       Anywhere (v6)             
# 32803/tcp (v6)             ALLOW       Anywhere (v6)             
# 7263 (v6)                  ALLOW       Anywhere (v6)             
# 7262 (v6)                  ALLOW       Anywhere (v6)             
# 8000/tcp (v6)              ALLOW       Anywhere (v6)             

# (base) umang@vrl-h200:~$ sudo ufw allow 22
# Rule added
# Rule added (v6)
# (base) umang@vrl-h200:~$ sudo ufw status
# Status: active

# To                         Action      From
# --                         ------      ----
# 22                         ALLOW       128.111.0.0/16            
# 2049                       ALLOW       128.111.56.136            
# 2049/tcp                   ALLOW       Anywhere                  
# 111/tcp                    ALLOW       Anywhere                  
# 111/udp                    ALLOW       Anywhere                  
# 20048/tcp                  ALLOW       Anywhere                  
# 875/tcp                    ALLOW       Anywhere                  
# 662/tcp                    ALLOW       Anywhere                  
# 32803/tcp                  ALLOW       Anywhere                  
# 7263                       ALLOW       Anywhere                  
# 7262                       ALLOW       Anywhere                  
# 8000/tcp                   ALLOW       Anywhere                  
# 22                         ALLOW       Anywhere                  
# 2049/tcp (v6)              ALLOW       Anywhere (v6)             
# 111/tcp (v6)               ALLOW       Anywhere (v6)             
# 111/udp (v6)               ALLOW       Anywhere (v6)             
# 20048/tcp (v6)             ALLOW       Anywhere (v6)             
# 875/tcp (v6)               ALLOW       Anywhere (v6)             
# 662/tcp (v6)               ALLOW       Anywhere (v6)             
# 32803/tcp (v6)             ALLOW       Anywhere (v6)             
# 7263 (v6)                  ALLOW       Anywhere (v6)             
# 7262 (v6)                  ALLOW       Anywhere (v6)             
# 8000/tcp (v6)              ALLOW       Anywhere (v6)             
# 22 (v6)                    ALLOW       Anywhere (v6)             

# (base) umang@vrl-h200:~$ sudo netstat -tuln | grep :22
# tcp6       0      0 :::22                   :::*                    LISTEN     
# (base) umang@vrl-h200:~$ sudo systemctl enable ssh
# Synchronizing state of ssh.service with SysV service script with /usr/lib/systemd/systemd-sysv-install.
# Executing: /usr/lib/systemd/systemd-sysv-install enable ssh
# Created symlink /etc/systemd/system/sshd.service → /usr/lib/systemd/system/ssh.service.
# Created symlink /etc/systemd/system/multi-user.target.wants/ssh.service → /usr/lib/systemd/system/ssh.service.
# (base) umang@vrl-h200:~$ 