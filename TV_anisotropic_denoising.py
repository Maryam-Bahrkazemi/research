import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from skimage.metrics import structural_similarity as ssim
import time
st = time.time()
####################functions:
def shrink(u, lambd):
    z = np.zeros(u.shape)
    b = np.sign(u) * np.maximum(np.abs(u) - (1. / lambd), z)
    return b
#####Gradient functions for x and y
def gradient_x(u):
    m, n = np.shape(u)
    u_x = np.zeros_like(u)
    u_x[0:m-1, :] = u[1:m, :] - u[0:(m - 1), :]
    u_x[m-1, :] = 0
    return u_x

def gradient_y(u):
    m, n = np.shape(u)
    u_y = np.zeros_like(u)
    u_y[:, 0:n-1] = u[:, 1:n] - u[:, 0:(n - 1)]
    u_y[:, n-1] = 0
    return u_y
#######gradient transpose functions for x and y
def gradient_t_x(u):
    m, n = np.shape(u)
    u_x = np.zeros_like(u)
    u_x[1:m, :] = u[0:m-1, :] - u[1:m, :]
    u_x[0, :] = -u[0, :]
    u_x[m-1, :] = u[m-2, :]
    return u_x

def gradient_t_y(u):
    m, n = np.shape(u)
    u_y = np.zeros_like(u)
    u_y[:, 1:n] = u[:, 0:n-1] - u[:, 1:n]
    u_y[:, 0] = -u[:, 0]
    u_y[:, n-1] = u[:, n-2]
    return u_y
##################
def l2_square(u):
    u = np.sum(np.square(np.abs(u)))
    return u

def psnr(u,f):
    q = -1 * np.log(np.divide(np.linalg.norm(u-f), np.linalg.norm(f , ord = np.inf)))
    return q

def tv_norm(u):
    u = np.sum(np.abs(gradient_x(u)) + np.abs(gradient_y(u)))
    return u
############################
######Reading the image and adding the gaussian noise
u_s = imread('Air_plane.tiff').astype(np.float64)
plt.imshow(u_s, cmap ='gray')
plt.title('original')
plt.colorbar()
plt.show()

np.random.seed(4589)
def gaussian(data, sigma):
    return data + np.random.normal(0.0, sigma, data.shape)
f = gaussian(u_s,15)

plt.imshow(f, cmap = 'gray')
plt.title("noisy_gaussian")
plt.colorbar()
plt.show()
######################Parameters
########number of iterations
n_iter = 500
########The parameters of algorithm
mu = 0.1
l = 2*mu
alf = 0.1
#######################Split Bregman algorithm for image denoising
u = f
bx = np.zeros_like(u_s, dtype=float)
dx = np.zeros_like(u_s, dtype=float)
by = np.zeros_like(u_s, dtype=float)
dy = np.zeros_like(u_s, dtype=float)
df = []
tv = []
for k in range(n_iter):
    #######u_sub problem using Gradient Descent algorithm:
    u = u - alf * (mu * (u - f) + l * (gradient_t_x(gradient_x(u)) - gradient_t_x(dx) + gradient_t_x(bx))
                   + l * (gradient_t_y(gradient_y(u)) - gradient_t_y(dy) + gradient_t_y(by)))

    grad_ux = gradient_x(u)
    grad_uy = gradient_y(u)

    #######d_sub problem using shrink function defined above
    dx = shrink(grad_ux + bx, l)
    dy = shrink(grad_uy + by, l)

    #######updating Bregman parameters
    bx += (grad_ux - dx)
    by += (grad_uy - dy)

    ####ploting
    tv.append(tv_norm(u))
    df.append(l2_square(u))

print('tv_original_image:', tv_norm(u_s))
print('tv_noisy_image:', tv_norm(f))
print('tv_denoised_image:', tv_norm(u))
print('PSNR_aniso_constraint', psnr(u,u_s))
print('SSIM_sb_constraint', ssim(u,u_s))

plt.imshow(u, cmap='gray')
plt.title('denoised')
plt.colorbar()
plt.show()

plt.subplot(1,2,1)
plt.plot(tv, label = 'tv')
plt.title('tv')
plt.subplot(1,2,2)
plt.plot(df, label = 'df')
plt.title('df')
plt.legend()
plt.show()

plt.plot(u[:, 128], label ='denoised')
plt.plot(f[:, 128], label ='noisy', alpha =0.5)
plt.plot(u_s[:, 128], label ='original', alpha =0.5)
plt.legend(loc ="upper right")
plt.show()

et = time.time()
res = et - st
print('Execution time:', res, 'sec')
