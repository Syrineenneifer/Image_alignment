Scanning:

I was presented with these cubes with different textures that we wanted to modelize numerically. More specifically, we wanted to scan one side of each cube in order to identify the various features that they presented.

After opening the camera and the gelsight application, I had to calibrate after cleaning the material and the glass in order to get the most accurate results possible.

Then, to keep track of the side that was scanned, we decided to place an x and y axes by putting stickers (blue for x and yellow for y) on the cubes.

I then placed the pad on top of the face that I wanted to scan (I usually chose the face that was the smoothest and with the most interesting features) and moved the camera until I saw features appearing on the computer screen.

We wanted to get 70 pictures for each cube, i.e. 7 scans per row. The scanning process consists of taking small images such that the ones next to each other have an overlapping region and we wanted that overlapping region each time to be approximately of the same width (for horizontal alignment) or height (for vertical alignment). Therefore, I decided to place stickers on the computer screen so I would more or less stop the manual sweeping around the same coordinate for each scan.

Once all 70 scans were taken, I had to generate the 3D model of each image to be able to obtain the heightmaps that we would use later as our data for the alignment.

Aligning:

The goal is to optimize the alignment between two scans, i.e. minimize a loss function based on the images provided. For that, we used the normal maps as the main tool to give us information about the pixels of the pictures. Therefore, given 2 close scans we start by sweeping through the first image (or the fixed image) starting from a specific offset, e.g. second half of the image, and we compare the first column of the area we’re going through with the first column or first row of the second image (or the moving image) since it is exactly the start of the common overlapping region.

NOTE THAT when the alignment is horizontal, we’re comparing the columns; when it’s a vertical alignment, we’re comparing the rows.

For the comparison, we firstly decided to use the squared difference of the two normal vectors, one from the first image and the other from the second image. Then, we tried to do one minus the dot product between both vectors and both methods give almost the same results.

We then plotted the loss function to be able to see a clear minimum and extract from that the value of the offset (the loss function also returns the best offset).

The shift number is added to the coordinate where we start sweeping in the picture and the sum gives the best offset for the alignment.

After that, I created a function that superimposes the moving image on top of the fixed image starting at the optimal offset.

NOTE THAT the images were scaled down by half in order to have a faster program, especially for row alignments.

I also created another alignment function based on the same principles but that uses cma. It was very buggy in the beginning because I had to adapt the sigma and the x0 values to each alignment which wasn’t an easy nor a fast process. Adding a noise handler and increasing the population size also helped in reducing the off and inadequate results.

The cma function returns more or less the same offset values as the standard function.

Cma has multiple parameters such as sigma. However, it was very tricky to find the adequate value for sigma because an inaccurate one gives off results such as negative or very small values. Therefore, for the horizontal alignment, sigma is equal to 0.15 * width and for the vertical alignment, sigma is equal to 0.06 * length.

Moreover, in order to visually compare the alignments given by the different offsets, I created a manual slider where you enter an offset in the range of the picture’s width then you click on a button and it computes the loss function (returns the value) and makes the alignment given by the chosen offset appear.
