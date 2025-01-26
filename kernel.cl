#define VIEWS_COUNT VIEW_COUNT_INPUT

float pixelDistance(float *a, float *b, int size)
{
    float dist = 0;
    for (int i=0; i<size; i++)
        dist += pow(a[i] - b[i], 2);
    dist = sqrt(dist);
    return dist;
}

float pixelDistanceVector(float4 a, float4 b)
{
    float4 fa = convert_float4(a);
    float4 fb = convert_float4(b);
    return pixelDistance((float*)&fa, (float*)&fb, 3);
}

float pixelDistanceScalar(float4 a, float4 b, int channel)
{
    float fa = a[channel];
    float fb = b[channel];
    return pixelDistance((float*)&fa, (float*)&fb, 1);
}

float3 getColorDepthVarianceAndAverageDepth(uint4 *pixels, int count)
{

    float4 averagePixel = (float4)(0, 0, 0, 0);
    for (int pixelID=0; pixelID<count; pixelID++)
        averagePixel = averagePixel + convert_float4(pixels[pixelID]);
    averagePixel /= count;

    float colorVariance = 0;
    float depthVariance = 0;
    for (int pixelID=0; pixelID<count; pixelID++)
    {
       colorVariance += pixelDistanceVector(convert_float4(pixels[pixelID]), averagePixel);
       depthVariance += pixelDistanceScalar(convert_float4(pixels[pixelID]), averagePixel, 3);
    }
    return (float3)(colorVariance/count, depthVariance/count, averagePixel[3]);  
}

#define NEIGHBORS 8
const float2 neighborhood[NEIGHBORS] = {
    (float2)(0.5f, 0.5f),
    (float2)(0.5f, -0.5f),
    (float2)(-0.5f, 0.5f),
    (float2)(-0.5f, -0.5f),
    (float2)(1.5f, 1.5f),
    (float2)(1.5f, -1.5f),
    (float2)(-1.5f, 1.5f),
    (float2)(-1.5f, -1.5f)
};

__kernel void kernelMain(__read_only image2d_t inputImage, __write_only image2d_t outputImage, int rows, int cols, float2 viewResolution, float2 halfPixel, float2 fullPixel, float2 viewRange, int applyLimits) 
{
    const sampler_t imageSampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
    int2 coords = (int2)(get_global_id(0), get_global_id(1)); 
    int2 inputResolution = (int2)(get_image_width(inputImage), get_image_height(inputImage));
    float2 coordsNormalized = (float2)((float)get_global_id(0)/(viewResolution[0]-1), (float)get_global_id(1)/(viewResolution[1]-1));

    if (coordsNormalized[0] > 1.0f || coordsNormalized[1] > 1.0f)
        return;
   
    float blockVariance = 0;
    uint4 pixels[NEIGHBORS][VIEWS_COUNT];
    for (int viewID=0; viewID<VIEWS_COUNT; viewID++)
    {
        int2 viewCoords = (int2)(viewID % cols, viewID / cols);
        float2 sampleCoords = viewRange*convert_float2(viewCoords) + coordsNormalized / (float2)(cols,rows);
        sampleCoords += halfPixel;
        for (int neighborID=0; neighborID<NEIGHBORS; neighborID++)
        {
            float2 currentSampleCoords = sampleCoords+(neighborhood[neighborID])*fullPixel;
            pixels[neighborID][viewID] = read_imageui(inputImage, imageSampler, currentSampleCoords);
        }
    }
    
    float colorVariance = 0;
    float depthVariance = 0;
    float averageDepth = 0;
    for (int neighborID=0; neighborID<NEIGHBORS; neighborID++)
    {
        float3 variances = getColorDepthVarianceAndAverageDepth(pixels[neighborID], VIEWS_COUNT);
        colorVariance += variances[0];
        depthVariance += variances[1];
        averageDepth += variances[2];
        float blockVariance = 0;
    }
    colorVariance /= NEIGHBORS;
    depthVariance /= NEIGHBORS;
    averageDepth /= NEIGHBORS;
    
    blockVariance = 0;    
    for (int viewID=0; viewID<VIEWS_COUNT; viewID++)
    {
        uint4 blockPixels[NEIGHBORS];
        for (int neighborID=0; neighborID<NEIGHBORS; neighborID++)
        {
            blockPixels[neighborID] = pixels[neighborID][viewID];
        }
        float3 variances = getColorDepthVarianceAndAverageDepth(blockPixels, NEIGHBORS);
        blockVariance += variances[0];  
    }
    blockVariance /= VIEWS_COUNT;

    float variance = colorVariance;
    if(applyLimits)
    {
        const float MAX_VAR = 255;
        if(depthVariance > 30 || averageDepth < 5 || blockVariance < 3)
            variance = MAX_VAR;
    }

	write_imagef(outputImage, coords, variance);
}
