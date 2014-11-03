from PIL import Image
import pyopencl as cl
import numpy
import os
from os.path import dirname
from os.path import join
from mako.template import Template
from unicodedata import normalize
from time import time

def main():
    image = Image.open('marbles.bmp')
    width, height = image.size

    cl = BlackAndWhite(image)


    cl.run()
    print("{} s, {}".format(cl.run_time, 0.066923867/cl.run_time))
    black_and_white = image.copy()
    #black_and_white = Image.new('RGB', (width, height), "black")

    for index in range(width * height):
        y = index / width
        x = index - (y * width)
        #print(width, height, x, y)
        black_and_white.putpixel((x, y), tuple(cl.output_pixel_array[index]))

    # print(list(black_and_white.getdata())[50000:50010])
    # print(cl.pixel_array[50000:50010])
    black_and_white.save("grey.bmp")


class BlackAndWhite(object):
    def __init__(self, original_image):
        super(BlackAndWhite, self).__init__()
        self.original_image = original_image
        self.pixel_array = numpy.array(list(self.original_image.getdata()))

        width, height = self.original_image.size
        self.total_size = width * height

        self.output_pixel_array = numpy.zeros((self.total_size, 3), dtype=numpy.int32)

    def setup_opencl(self):
        # Setup context, queue
        # Set to use GPU
        platform = cl.get_platforms()
        my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        # for device in my_gpu_devices:
            # print(device)

        self.cl_context = cl.Context(devices=my_gpu_devices)
        self.cl_queue = cl.CommandQueue(self.cl_context, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # Setup buffers
        mf = cl.mem_flags

        self.cl_input_image_buf = cl.Buffer(self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pixel_array)

        self.cl_output_image_buf = cl.Buffer(self.cl_context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.output_pixel_array)

        # Get kernel, send parameters for the mako file
        to_black_and_white = self.get_kernel("to_black_and_white")

        self.cl_program = cl.Program(self.cl_context, to_black_and_white).build()

    def run(self):
        self.setup_opencl()

        global_size = (self.total_size, )
        # local_size = (256, )
        local_size = None
        start_time = time()

        compute_event = self.cl_program.to_black_and_white(self.cl_queue, global_size, local_size, self.cl_input_image_buf, self.cl_output_image_buf)
        compute_event.wait()
        cl.enqueue_read_buffer(self.cl_queue, self.cl_output_image_buf, self.output_pixel_array).wait()

        end_time = time()

        self.run_time = 1e-9*(compute_event.profile.end - compute_event.profile.start)

    def get_kernel(self, file_name, **parameters):
        # get current directory, look in kernels/ for the mako file
        path = join(dirname(__file__), 'kernels/{}.mako'.format(file_name))
        with open(path, 'r') as kernel_file:
            text = kernel_file.read()
            kernel = Template(text).render(**parameters)
            # opencl returns a warning if the code in unicode, so convert it
            return normalize('NFKD', kernel).encode('ascii','ignore')


if __name__ == '__main__':
    main()