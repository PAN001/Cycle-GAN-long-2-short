import matplotlib.pyplot as plt
import pylab as pl
from operator import add

filename = "./log_skin_seg2.out"
file = open(filename, "r")
cnt = 0
gan_A_losses = []
d_A_losses = [] #D_X
gan_B_losses = []
d_B_losses = []
for line in file:
    if cnt == 0:
        cnt += 1
        continue

    splitted = line.split(",")
    print splitted

    G_A = float(splitted[0])
    gan_A_losses.append(G_A)
    D_A = float(splitted[1])
    d_A_losses.append(D_A)

    G_B = float(splitted[2])
    gan_B_losses.append(G_B)
    D_B = float(splitted[3])
    d_B_losses.append(D_B)

    cnt += 1

fig = plt.figure(2, figsize=(40, 10))

# # GAN loss
# plt.plot(gan_A_losses)
# plt.plot(gan_B_losses)
#
#
# plt.title('Cycle-GAN shortpants2longpants: GAN loss vs. iter', fontsize=28)
# plt.ylabel('GAN loss', fontsize=18)
# plt.xlabel('iter', fontsize=18)
# plt.legend(['GAN_A loss', 'GAN_B loss'], loc='upper left')
# # plt.show()
# plt.savefig("./plots/GAN_loss.png")

# d loss
plt.plot(d_A_losses)
plt.plot(d_B_losses)

plt.title('Cycle-GAN shortpants2longpants: Discriminator loss vs. iter', fontsize=28)
plt.ylabel('Discriminator loss', fontsize=18)
plt.xlabel('iter', fontsize=18)
plt.legend(['D_A loss', 'D_B loss'], loc='upper left')
# plt.show()
plt.savefig("./plots/discriminator_loss.png")