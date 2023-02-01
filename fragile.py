"""
This program was made for the data security course of Karadeniz Technical University.

Uğurcan Yılmaz
383242@ogr.ktu.edu.tr
Karadeniz Technical University
"""
import rsa
import ast
import Crypto
import hashlib
from math import log10, sqrt
import random
import sys
import cv2
import numpy as np
from PIL import Image
from bitstring import BitArray

def XOR(bit1, bit2):
    if bit1 == bit2:
        return 0
    else:
        return 1

def insert_watermark(host_image, watermark_image, key_size=64):
    height, width = host_image.shape[:2]

    watermark_image = cv2.resize(watermark_image, (width, height))

    for i in range(watermark_image.shape[0]):
            for j in range(watermark_image.shape[1]):
                if watermark_image[i][j] > 127:
                    watermark_image[i][j] = 1
                else:
                    watermark_image[i][j] = 0

    block_size = 8

    sub_image_x = width // block_size
    sub_image_y = height // block_size

    host_blocks = []
    watermark_blocks = []
    host_blocks_lsb = []
    watermarked_blocks = []

    for i in range(sub_image_y):
        for j in range(sub_image_x):
            x0 = j * block_size
            y0 = i * block_size
            x1 = x0 + block_size
            y1 = y0 + block_size

            sub_image = host_image[y0:y1, x0:x1]
            host_blocks.append(sub_image)

            sub_image = watermark_image[y0:y1, x0:x1]
            watermark_blocks.append(sub_image)

    host_blocks_lsb = host_blocks

    for host_block_lsb, watermark_block in zip(host_blocks_lsb, watermark_blocks):
        for i in range(host_block_lsb.shape[0]):
            for j in range(host_block_lsb.shape[1]):
                if (host_block_lsb[i][j] % 2) != 0:
                    host_block_lsb[i][j] -= 1
        
        host_block_bytes = host_block_lsb.tobytes()
        m = hashlib.md5()
        m.update(host_block_bytes)
        hash = m.digest()
        host_block_hash_bits = ''.join(f'{b:08b}' for b in hash)
        
        first_64_bit_of_hash = host_block_hash_bits[:64]

        flattened_watermark_block = watermark_block.flatten()

        XOR_of_hash_and_watermark = []
        for i in range(64):
            XOR_of_hash_and_watermark.append(XOR(flattened_watermark_block[i], int(first_64_bit_of_hash[i])))
        
        """ 
        public_key, private_key = rsa.newkeys(2048)

        with open("keys/public.pem", "wb") as f:
            f.write(public_key.save_pkcs1("PEM"))
        
        with open ("keys/private.pem", "wb") as f:
            f.write(private_key.save_pkcs1("PEM"))
        

        with open("keys/public.pem", "rb") as f:
            public_key = rsa.PublicKey.load_pkcs1(f.read())
        
        with open("keys/private.pem", "rb") as f:
            private_key = rsa.PrivateKey.load_pkcs1(f.read())

        encrypted_XOR_of_hash_and_watermark = rsa.encrypt(XOR_of_hash_and_watermark.encode(), public_key)
        with open("message/encrypted.message", "wb") as f:
            f.write(encrypted_XOR_of_hash_and_watermark)
        """
        
        XOR_of_hash_and_watermark_array = np.reshape(XOR_of_hash_and_watermark, (8, 8))

        for i in range(8):
            for j in range(8):
                if XOR_of_hash_and_watermark_array[i][j] == 1:
                    host_block_lsb[i][j] += 1

        watermarked_blocks.append(host_block_lsb)
    
    watermarked_image = np.zeros((host_image.shape[0], host_image.shape[1]), dtype=np.uint8)

    k = 0
    for i in range(sub_image_y):
        for j in range(sub_image_x):
            x0 = j * block_size
            y0 = i * block_size
            x1 = x0 + block_size
            y1 = y0 + block_size

            watermarked_image[y0:y1, x0:x1] = watermarked_blocks[k]
            k += 1

    return watermarked_image

def extract_watermark(watermarked_image):
    height, width = watermarked_image.shape[:2]

    block_size = 8

    sub_image_x = width // block_size
    sub_image_y = height // block_size


    watermarked_blocks = []
    extracted_watermark_blocks = []

    for i in range(sub_image_y):
        for j in range(sub_image_x):
            x0 = j * block_size
            y0 = i * block_size
            x1 = x0 + block_size
            y1 = y0 + block_size

            sub_image = watermarked_image[y0:y1, x0:x1]
            watermarked_blocks.append(sub_image)

    watermarked_blocks_lsb = watermarked_blocks
    watermark_blocks = []

    for  watermarked_block_lsb, watermarked_block in zip(watermarked_blocks_lsb, watermarked_blocks):
        lsbs_of_watermarked = []
        for i in range(watermarked_block_lsb.shape[0]):
            for j in range(watermarked_block_lsb.shape[1]):
                if (watermarked_block_lsb[i][j] % 2) != 0:
                    lsbs_of_watermarked.append(1)
                    watermarked_block_lsb[i][j] -= 1
                else:
                    lsbs_of_watermarked.append(0)

        watermarked_block_bytes = watermarked_block_lsb.tobytes()
        m = hashlib.md5()
        m.update(watermarked_block_bytes)
        hash = m.digest()
        watermarked_block_hash_bits = ''.join(f'{b:08b}' for b in hash)
        
        first_64_bit_of_hash = watermarked_block_hash_bits[:64]

        XOR_of_hash_and_watermark = []
        for i in range(64):
            XOR_of_hash_and_watermark.append(XOR(lsbs_of_watermarked[i], int(first_64_bit_of_hash[i])))
        
        watermark_block = XOR_of_hash_and_watermark

        watermark_block = np.reshape(watermark_block, (8, 8))

        for i in range(8):
            for j in range(8):
                if watermark_block[i][j] == 1:
                    watermark_block[i][j] = 255
        

        

        watermark_blocks.append(watermark_block)
    
    extracted_watermark_image = np.zeros((watermarked_image.shape[0], watermarked_image.shape[1]), dtype=np.uint8)

    k = 0
    for i in range(sub_image_y):
        for j in range(sub_image_x):
            x0 = j * block_size
            y0 = i * block_size
            x1 = x0 + block_size
            y1 = y0 + block_size

            extracted_watermark_image[y0:y1, x0:x1] = watermark_blocks[k]
            k += 1

    return extracted_watermark_image

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr




"""
-Host image and watermark image can be any size.
-This program currently only accepts grayscale images, but I will add color image support in the future.
    (idk when but i will)
-If you use color images it converts them to grayscale and then uses them.
-Run the block in the below to use the watermark insertion function.
    (don't forget to add host and watermark image paths)
"""

""" ##uncomment this block##
host_image = cv2.imread('host_image_path', 0)
watermark_image = cv2.imread('watermark_image_path', 0)

watermarked_image = insert_watermark(host_image, watermark_image)
cv2.imwrite('watermarked_image.png', watermarked_image)
"""



"""
-draw a line on the watermarked image and try to extract watermark image from manipulated image. 
    You can examine the manipulation on the extracted watermark.
-First comment the insertion block and then run the block in the below to use the watermark extraction function.
    (don't forget to add watermarked image path)
"""

""" ##uncomment this block##
watermarked_image_2 = cv2.imread('watermarked_image.png',0)

extracted_watermark = extract_watermark(watermarked_image_2)
cv2.imwrite('extracted_watermark.jpg', extracted_watermark)
"""

#value = PSNR(host_image, watermarked_image_2)
#print(f"PSNR value is {value} dB")