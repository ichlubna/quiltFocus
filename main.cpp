#include <CL/opencl.hpp>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <exception>
#include "arguments/arguments.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

constexpr int imageChannelsGPU = 4;

class Params
{
    public:
    std::string inputPath;
    std::string outputPath;
    int cols;
    int rows;
};

void storeGPUImage(cl::CommandQueue queue, cl::Image2D image, std::string path, bool imageFloat=false)
{
    size_t width{0}, height{0};
    image.getImageInfo(CL_IMAGE_WIDTH, &width);
    image.getImageInfo(CL_IMAGE_HEIGHT, &height);
    if (imageFloat)
    {
        std::vector<float> outData;
        outData.resize(width * height);
        if(queue.enqueueReadImage(image, CL_TRUE, cl::array<size_t, 3>{0, 0, 0}, cl::array<size_t, 3>{static_cast<size_t>(width), static_cast<size_t>(height), 1}, 0, 0, outData.data()) != CL_SUCCESS)
            throw std::runtime_error("Cannot download the result");
        stbi_write_hdr(path.c_str(), width, height, 1, outData.data());
    }
    else
    {
        std::vector<unsigned char> outData;
        outData.resize(width * height * imageChannelsGPU);
        if(queue.enqueueReadImage(image, CL_TRUE, cl::array<size_t, 3>{0, 0, 0}, cl::array<size_t, 3>{static_cast<size_t>(width), static_cast<size_t>(height), 1}, 0, 0, outData.data()) != CL_SUCCESS)
            throw std::runtime_error("Cannot download the result");
        stbi_write_png(path.c_str(), width, height, imageChannelsGPU, outData.data(), width * imageChannelsGPU);
    }
}

void process(Params params)
{
    std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0)
        throw std::runtime_error("No OpenCL platforms available");
    cl::Context context(CL_DEVICE_TYPE_DEFAULT);
    std::ifstream file("kernel.cl");
    std::stringstream kernelContent;
    kernelContent << file.rdbuf();
    file.close();
    std::string kernelString = kernelContent.str();
    kernelString.replace(kernelString.find("VIEW_COUNT_INPUT"), sizeof("VIEW_COUNT_INPUT") - 1, std::to_string(params.cols * params.rows));
    cl::Program program(context, kernelString, true);
    cl::CommandQueue queue(context);

    std::cerr << "Loading images and allocating GPU memory" << std::endl;
    const cl::ImageFormat imageFormat(CL_RGBA, CL_UNSIGNED_INT8);
    unsigned char *imageData{nullptr}; 
    int viewCount = params.cols * params.rows;
    int imageWidth{0}, imageHeight{0}, imageChannels{0};
    int viewWidth{0}, viewHeight{0}, viewChannels{0};
    if (std::filesystem::is_directory(params.inputPath))
    {
        for (const auto& file : std::filesystem::directory_iterator(params.inputPath))
        {
            unsigned char *imageData = stbi_load(file.path().c_str(), &viewWidth, &viewHeight, &viewChannels, imageChannelsGPU);
            if (imageData == nullptr)
                throw std::runtime_error("Failed to load image");
            break;
        }
        imageWidth = viewWidth*params.cols;
        imageHeight = viewHeight*params.rows;
    }
    else
    {
        imageData = stbi_load(params.inputPath.c_str(), &imageWidth, &imageHeight, &imageChannels, imageChannelsGPU);
        if (imageData == nullptr)
            throw std::runtime_error("Failed to load image " + params.inputPath);
    }
   
	cl::Image2D inputImageGPU(context, CL_MEM_READ_ONLY, imageFormat, imageWidth, imageHeight, 0, nullptr);
    
    if (std::filesystem::is_directory(params.inputPath))
    {
        int counter = 0;
        auto iterator = std::filesystem::directory_iterator(params.inputPath);
        std::vector<std::filesystem::path> files; 
        for (const auto& file : std::filesystem::directory_iterator(params.inputPath))
            files.push_back(file);
        std::sort(files.begin(), files.end());
        for (const auto& file : files)
        {
            imageData = stbi_load(file.c_str(), &viewWidth, &viewHeight, &viewChannels, imageChannelsGPU);
            if (imageData == nullptr)
                throw std::runtime_error("Failed to load image " + file.string());
            cl::array<size_t, 3> origin{0, 0, 0};
            int x = counter % params.cols;
            int y = params.rows - 1 - (counter / params.cols);
            origin[0] = x*viewWidth;
            origin[1] = y*viewHeight;
            cl::array<size_t, 3> size{static_cast<size_t>(viewWidth), static_cast<size_t>(viewHeight), 1};
            if(queue.enqueueWriteImage(inputImageGPU, CL_TRUE, origin, size, 0, 0, imageData) != CL_SUCCESS)
                throw std::runtime_error("Cannot upload the image " + file.string() + " to GPU");
            stbi_image_free(imageData);
            counter++;
            if(counter > viewCount)
            {
                std::cerr << "The number of input files is higher than the expected quilt size. Using only the first " << viewCount << " files" << std::endl;
                break;
            }
        }
        if(counter < viewCount-1)
            throw std::runtime_error("The number of input images is lower than the expected quilt size");
        std::cerr << "Storing the quilt" << std::endl;
        storeGPUImage(queue, inputImageGPU, std::filesystem::path(params.outputPath) / "quilt.png");
    }
    else
    {
        if(queue.enqueueWriteImage(inputImageGPU, CL_TRUE, cl::array<size_t, 3>{0, 0, 0}, cl::array<size_t, 3>{static_cast<size_t>(imageWidth), static_cast<size_t>(imageHeight), 1}, 0, 0, imageData) != CL_SUCCESS)
            throw std::runtime_error("Cannot upload the quilt to GPU");
        stbi_image_free(imageData);
    }
         
    const cl::ImageFormat outputImageFormat(CL_R, CL_FLOAT);
	cl::Image2D outputImageGPU(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, outputImageFormat, viewWidth, viewHeight, 0, nullptr);

    std::cerr << "Processing on GPU" << std::endl;
    auto kernel = cl::compatibility::make_kernel<cl::Image2D&,cl::Image2D&, int, int>(program, "kernelMain"); 
    cl_int buildErr = CL_SUCCESS; 
    auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
    for (auto &pair : buildInfo)
        if(!pair.second.empty() && !std::all_of(pair.second.begin(),pair.second.end(),isspace))
            std::cerr << pair.second << std::endl;
    cl::EnqueueArgs kernelArgs(queue, cl::NDRange(viewWidth, viewHeight));
    kernel(kernelArgs, inputImageGPU, outputImageGPU, params.rows, params.cols);
    queue.finish();

    std::cerr << "Storing the result" << std::endl;
    storeGPUImage(queue, outputImageGPU, std::filesystem::path(params.outputPath) / "output.hdr", true);
}

int main(int argc, char *argv[])
{
    std::string helpText =  "This program takes a quilt image and produces the native Looking Glass image. All parameters below need to be specified according to the display model.\n"
                            "--help, -h Prints this help\n"
                            "-i input quilt image or directory - 8-BIT RGBA, all views having the same resolution\n"
                            "-o output directory - results stored as output.png and quilt.png\n"
                            "-rows number of rows in the quilt\n"
                            "-cols number of cols in the quilt\n";
    Arguments args(argc, argv);
    if(args.printHelpIfPresent(helpText))
        return 0;
    if(argc < 2)
    {
        std::cerr << "Use --help" << std::endl;
        return 0;
    }

    Params params;
    params.inputPath = static_cast<std::string>(args["-i"]);
    params.outputPath = static_cast<std::string>(args["-o"]);
    params.cols = static_cast<int>(args["-cols"]);
    params.rows = static_cast<int>(args["-rows"]);

    try
    {
        process(params);
    }

    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
