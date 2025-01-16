#define VIEWS_COUNT VIEW_COUNT_INPUT

__kernel void kernelMain(__read_only image2d_t inputImage, __write_only image2d_t outputImage, int rows, int cols) 
{
    const sampler_t imageSampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
    int2 coords = (int2)(get_global_id(0), get_global_id(1)); 
    float2 coordsNormalized = (float2)((float)get_global_id(0)/(get_image_width(outputImage)-1), (float)get_global_id(1)/(get_image_height(outputImage)-1));

    if (coordsNormalized[0] > 1.0f || coordsNormalized[1] > 1.0f)
        return;

    uint3 pixels[VIEWS_COUNT];
    for (int viewID=0; viewID<VIEWS_COUNT; viewID++)
    {
        int2 viewCoords = (int2)(viewID % cols, viewID / cols);
        float2 viewRange = (float2)(1.0f/cols, 1.0f/rows);
        float2 sampleCoords = (float2)(viewRange[0]*viewCoords[0] + coordsNormalized[0]/cols, viewRange[1]*viewCoords[1] + coordsNormalized[1]/rows);
        uint4 pixel = (read_imageui(inputImage, imageSampler, sampleCoords));
        pixels[viewID] = (uint3)(pixel[0], pixel[1], pixel[2]);
    }
    float3 averagePixel = (float3)(0, 0, 0);
    for (int viewID=0; viewID<VIEWS_COUNT; viewID++)
        averagePixel = (float3)(averagePixel[0] + pixels[viewID][0], averagePixel[1] + pixels[viewID][1], averagePixel[2] + pixels[viewID][2]);
    averagePixel = (float3)(averagePixel[0]/VIEWS_COUNT, averagePixel[1]/VIEWS_COUNT, averagePixel[2]/VIEWS_COUNT);
    float variance = 0;
    for (int viewID=0; viewID<VIEWS_COUNT; viewID++)
       variance += pow(pixels[viewID][0] - averagePixel[0], 2) + pow(pixels[viewID][1] - averagePixel[1], 2) + pow(pixels[viewID][2] - averagePixel[2], 2);
    variance /= VIEWS_COUNT; 
	write_imagef(outputImage, coords, variance);
}
