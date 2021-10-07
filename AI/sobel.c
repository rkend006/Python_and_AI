#include <stdio.h>                          /* Sobel.c */
#include <math.h>
#include <stdlib.h>

int pic[256][256];
int outpicx[256][256];
int outpicy[256][256];
int maskx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
int masky[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};
int xyConvolution[2] = { 1, -1 };
double ival[256][256],tHoldHI[256][256], tHoldLO[256][256], maxival;


int main(argc, argv)
int argc;
char **argv;
{
    int i,j,p,q,mr,sum1,sum2;
    double threshold;
    FILE *fp1, *fo1, *fo2, *fo3, *fopen();
    char *foobar;

    // Code for command-line programming.
    argc--; argv++; foobar = *argv; fp1=fopen(foobar,"rb");

	argc--; argv++; foobar = *argv; fo1=fopen(foobar,"wb");

    argc--; argv++; foobar = *argv; fo2=fopen(foobar,"wb");

    argc--; argv++; foobar = *argv; fo3=fopen(foobar,"wb");

    argc--; argv++; foobar = *argv; threshold = atof(foobar);

    // This was necessary to read in image values that did not matter.
    // Without doing so would create the image to be cutoff.
    for (i = 0; i < 40; i++)
        getc(fp1);

    // Reads in the image.
    for (i=0;i<256;i++){
        for (j=0;j<256;j++){
            
            pic[i][j]  =  getc (fp1);
            pic[i][j]  &= 0377;
            tHoldHI[i][j] = 0;
            tHoldLO[i][j] = 0;
        }
    }

    // Goes around one pixel in each direction for a total of 8 neighboring pixels.
    mr = 1;
    for (i=mr;i<256-mr;i++){
        for (j=mr;j<256-mr;j++){
            sum1 = 0;
            sum2 = 0;

            for (p=-mr;p<=mr;p++){
                for (q=-mr;q<=mr;q++){
                sum1 += pic[i+p][j+q] * maskx[p+mr][q+mr];
                sum2 += pic[i+p][j+q] * masky[p+mr][q+mr];
                }
            }
            outpicx[i][j] = sum1;
            outpicy[i][j] = sum2;
        }
    }

    maxival = 0;
    for (i=mr;i<256-mr;i++){
        for (j=mr;j<256-mr;j++){
            ival[i][j]=sqrt((double)((outpicx[i][j]*outpicx[i][j]) +
                                      (outpicy[i][j]*outpicy[i][j])));
            if (ival[i][j] > maxival)
                maxival = ival[i][j];
        }
    }

    // These create the three new images.
    fprintf(fo1,"P5\n");
    fprintf(fo1,"%d %d\n", 256, 256);
    fprintf(fo1,"255\n");

    fprintf(fo2,"P5\n");
    fprintf(fo2,"%d %d\n", 256, 256);
    fprintf(fo2,"255\n");

    fprintf(fo3,"P5\n");
    fprintf(fo3,"%d %d\n", 256, 256);
    fprintf(fo3,"255\n");
    
    int diff = 0, HI = 240, LO = 80;
    for (i=0;i<256;i++){
        for (j=0;j<256;j++){
  

            ival[i][j] = (ival[i][j] / maxival) * 255;
            fprintf(fo1,"%c",(char)((int)(ival[i][j])));

            if (i == 255)
                diff = (ival[i][j] * xyConvolution[0]) + (ival[i - 1][j] * xyConvolution[1]);
            else
                diff = (ival[i + 1][j] * xyConvolution[0]) + (ival[i][j] * xyConvolution[1]);

            if (abs(diff) > LO) {
                tHoldLO[i][j] = 255;
            }
            if (abs(diff) > HI) {
                tHoldHI[i][j] = 255;
            }
            fprintf(fo2, "%c", (char)((int)(tHoldLO[i][j])));
            fprintf(fo3, "%c", (char)((int)(tHoldHI[i][j])));
        }
    }

    printf("\n*************************************************************\n");
    printf("*                                                           *\n");
    printf("*  SCANNING IMAGE...                                        *\n");
    printf("*                                                           *\n");
    printf("*  THE IMAGE IS IN YOUR ROBOT VISION FOLDER.                *\n");
    printf("*                                                           *\n");
    printf("*************************************************************\n");

    return 0;
}
