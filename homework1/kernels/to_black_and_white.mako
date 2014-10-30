#define RED 0
#define GREEN 1
#define BLUE 2

__kernel void to_black_and_white(__global int *input_image_buf, __global int *output_image_buf)
{
  int gid = get_global_id(0);
  int pixel_index = gid * 3;

  int red = input_image_buf[pixel_index + RED];
  int green = input_image_buf[pixel_index + GREEN];
  int blue = input_image_buf[pixel_index + BLUE];
  int intensity =  .299f * red + .587f * green + .114f * blue;

  output_image_buf[pixel_index + RED] = intensity;
  output_image_buf[pixel_index + GREEN] = intensity;
  output_image_buf[pixel_index + BLUE] = intensity;
}