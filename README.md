# digital-watermarking
This project is for protecting the data authenticity of images.
I used Ping Wah Wong's algorithm for watermark insertion and extraction.[1]
The only difference of the program from the Wong's algorithm is that it does not include the public key encryption part but I will add that part in the future.

Below image shows the watermark insertion algorithm:
![image](https://user-images.githubusercontent.com/72527510/216027483-3db4a277-a156-493a-a684-a5f21a56221c.png)


Below image shows the watermark extraction algorithm:
![image](https://user-images.githubusercontent.com/72527510/216027968-7085f3f8-59c0-4e2e-a773-6dcd9df0cbf2.png)

host image: 

![host](https://user-images.githubusercontent.com/72527510/216032234-71cde140-1f02-41bc-8525-c6123f1e3e5a.jpg)

watermark image: 

![watermark](https://user-images.githubusercontent.com/72527510/216032352-88fa2fbd-47e4-41dd-9a59-a0f6eaf76ce1.jpg)

watermarked image:

![watermarked_image](https://user-images.githubusercontent.com/72527510/216032605-25269680-dbb7-4c9a-96b1-34c8cff8e8a5.png)

manipulated watermarked image: (numbers on the receipt are manipulated)

![watermarked_image](https://user-images.githubusercontent.com/72527510/216033692-fa6c674c-a5c1-4bbd-9010-2fcbbb1f80d6.png)

extracted watermark image: (shows the manipulations)

![extracted_watermark](https://user-images.githubusercontent.com/72527510/216034286-5b9e8409-8951-466a-99c3-98053c989c5e.jpg)

Check the code for more information.

## references
[1] P. W. Wong, “A Public Key Watermark for Image Verification and 
Authentication” May 1998. (https://ieeexplore.ieee.org/document/723526)