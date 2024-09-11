#include "io/IoArray.h"
#include "transform/TransArray.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace skmap;

dict_t convPyDict(py::dict in_dict)
{
    dict_t cpp_dict;
    for (auto item : in_dict) {
        cpp_dict[py::str(item.first)] = py::str(item.second);
    }
    return cpp_dict;
}


map_t convPyMap(py::dict in_map)
{
    map_t cpp_map;
    for (auto item : in_map) {
        cpp_map[py::str(item.first)] = item.second.cast<std::vector<uint_t>>();
    }
    return cpp_map;
}


void readData(Eigen::Ref<MatFloat> data,
              const uint_t n_threads,
              const std::vector<std::string>& file_locs,
              const std::vector<uint_t> perm_vec,
              const uint_t x_off,
              const uint_t y_off,
              const uint_t x_size,
              const uint_t y_size,
              const std::vector<int> bands_list,
              py::dict conf_GDAL,
              std::optional<float_t> value_to_mask,
              std::optional<float_t> value_to_set) 
{
    IoArray ioArray(data, n_threads);
    ioArray.setupGdal(convPyDict(conf_GDAL));
    ioArray.readData(file_locs, perm_vec, x_off, y_off, x_size, y_size, GDALDataType::GDT_Float32,
                     bands_list, value_to_mask, value_to_set);
}

void getLatLonArray(Eigen::Ref<MatFloat> data,
                    const uint_t n_threads,
                    py::dict conf_GDAL,
                    std::string file_loc,
                    uint_t x_off,
                    uint_t y_off,
                    uint_t x_size,
                    uint_t y_size)
{
    IoArray ioArray(data, n_threads);
    ioArray.setupGdal(convPyDict(conf_GDAL));
    ioArray.getLatLonArray(file_loc, x_off, y_off, x_size, y_size);
}



void reorderArray(Eigen::Ref<MatFloat> data,
                  const uint_t n_threads,
                  Eigen::Ref<MatFloat> out_data,
                  std::vector<std::vector<uint_t>> indices_matrix)
{
    TransArray transArray(data, n_threads);
    transArray.reorderArray(out_data, indices_matrix);
}



void selArrayRows(Eigen::Ref<MatFloat> data,
                  const uint_t n_threads,
                  Eigen::Ref<MatFloat> out_data,
                  std::vector<uint_t> row_select)
{
    TransArray transArray(data, n_threads);
    transArray.selArrayRows(out_data, row_select);
}



void testLib()
{
    std::cout << "Ciaoooo!" << std::endl;
}

void expandArrayRows(Eigen::Ref<MatFloat> data,
                     const uint_t n_threads,
                     Eigen::Ref<MatFloat> out_data,
                     std::vector<uint_t> row_select)
{
    TransArray transArray(data, n_threads);
    transArray.expandArrayRows(out_data, row_select);
}

void extractArrayRows(Eigen::Ref<MatFloat> data,
                     const uint_t n_threads,
                     Eigen::Ref<MatFloat> out_data,
                     std::vector<uint_t> row_select)
{
    TransArray transArray(data, n_threads);
    transArray.extractArrayRows(out_data, row_select);
}

void swapRowsValues(Eigen::Ref<MatFloat> data,
                    const uint_t n_threads,
                    std::vector<uint_t> row_select,
                    float_t value_to_mask,
                    float_t new_value)
{
    TransArray transArray(data, n_threads);
    transArray.swapRowsValues(row_select, value_to_mask, new_value);
}

void maskNan(Eigen::Ref<MatFloat> data,
                    const uint_t n_threads,
                    std::vector<uint_t> row_select,
                    float_t new_value_in_data)
{
    TransArray transArray(data, n_threads);
    transArray.maskNan(row_select, new_value_in_data);
}

void maskData(Eigen::Ref<MatFloat> data,
                    const uint_t n_threads,
                    std::vector<uint_t> row_select,
                    Eigen::Ref<MatFloat> mask,
                    float_t value_of_mask_to_mask,
                    float_t new_value_in_data)
{
    TransArray transArray(data, n_threads);
    transArray.maskData(row_select, mask, value_of_mask_to_mask, new_value_in_data);
}

void fillArray(Eigen::Ref<MatFloat> data,
               const uint_t n_threads,
               float_t val)
{
    TransArray transArray(data, n_threads);
    transArray.fillArray(val);
}

void copyVecInMatrixRow(Eigen::Ref<MatFloat> data,
               const uint_t n_threads,
               Eigen::Ref<VecFloat> in_vec,
               uint_t row_idx)
{
    TransArray transArray(data, n_threads);
    transArray.copyVecInMatrixRow(in_vec, row_idx);
}


void inverseReorderArray(Eigen::Ref<MatFloat> data,
                          const uint_t n_threads,
                          Eigen::Ref<MatFloat> out_data,
                          std::vector<std::vector<uint_t>> indices_matrix)
{
    TransArray transArray(data, n_threads);
    transArray.inverseReorderArray(out_data, indices_matrix);
}

void transposeReorderArray(Eigen::Ref<MatFloat> data,
                         const uint_t n_threads,
                         Eigen::Ref<MatFloat> out_data,
                         std::vector<std::vector<uint_t>> permutation_matrix)
{
    TransArray transArray(data, n_threads);
    transArray.transposeReorderArray(out_data, permutation_matrix);
}

void transposeArray(Eigen::Ref<MatFloat> data,
                          const uint_t n_threads,
                          Eigen::Ref<MatFloat> out_data)
{
    TransArray transArray(data, n_threads);
    transArray.transposeArray(out_data);
}

void offsetAndScale(Eigen::Ref<MatFloat> data,
                    const uint_t n_threads,
                    float_t offset,
                    float_t scaling)
{
    TransArray transArray(data, n_threads);
    transArray.offsetAndScale(offset, scaling);
}

void scaleAndOffset(Eigen::Ref<MatFloat> data,
                    const uint_t n_threads,
                    float_t offset,
                    float_t scaling)
{
    TransArray transArray(data, n_threads);
    transArray.scaleAndOffset(offset, scaling);
}


void fitPercentage(Eigen::Ref<MatFloat> out,
                   const uint_t n_threads,
                   Eigen::Ref<MatFloat> in1,
                   Eigen::Ref<MatFloat> in2)
{
    TransArray transArray(out, n_threads);
    transArray.fitPercentage(in1, in2);
}



void averageAi4sh(Eigen::Ref<MatFloat> out,
                  const uint_t n_threads,
                  Eigen::Ref<MatFloat> in1,
                  Eigen::Ref<MatFloat> in2,
                  uint_t n_pix,
                  uint_t y)
{
    TransArray transArray(out, n_threads);
    transArray.averageAi4sh(in1, in2, n_pix, y);
}

void hadamardProduct(Eigen::Ref<MatFloat> out,
                     const uint_t n_threads,
                     Eigen::Ref<MatFloat> in1,
                     Eigen::Ref<MatFloat> in2)
{
    TransArray transArray(out, n_threads);
    transArray.hadamardProduct(in1, in2);
}


void nanMean(Eigen::Ref<MatFloat> data,
             const uint_t n_threads,
             Eigen::Ref<VecFloat> out_data)
{
    TransArray transArray(data, n_threads);
    transArray.nanMean(out_data);
}

void linearRegression(Eigen::Ref<MatFloat> data,
                      const uint_t n_threads,
                      Eigen::Ref<VecFloat> x,
                      Eigen::Ref<VecFloat> beta_0,
                      Eigen::Ref<VecFloat> beta_1)
{
    TransArray transArray(data, n_threads);
    transArray.linearRegression(x, beta_0, beta_1);
}


void computeMannKendallPValues(Eigen::Ref<MatFloat> data,
                      const uint_t n_threads,
                      Eigen::Ref<VecFloat> out_data)
{
    TransArray transArray(data, n_threads);
    transArray.computeMannKendallPValues(out_data);
}

void averageAggregate(Eigen::Ref<MatFloat> data,
                      const uint_t n_threads,
                      Eigen::Ref<MatFloat> out_data,
                      uint_t agg_factor)
{
    TransArray transArray(data, n_threads);
    transArray.averageAggregate(out_data, agg_factor);
}

void maskDifference(Eigen::Ref<MatFloat> data,
                    const uint_t n_threads,
                    float_t diff_th,
                    uint_t count_th,
                    Eigen::Ref<MatFloat> ref_data,
                    Eigen::Ref<MatFloat> mask_out)
{
    TransArray transArray(data, n_threads);
    transArray.maskDifference(diff_th, count_th, ref_data, mask_out);
}

void computeNormalizedDifference(Eigen::Ref<MatFloat> data,
                                 const uint_t n_threads,
                                 std::vector<uint_t> positive_indices,
                                 std::vector<uint_t> negative_indices,
                                 std::vector<uint_t> result_indices,
                                 float_t positive_scaling,
                                 float_t negative_scaling,
                                 float_t result_scaling,
                                 float_t result_offset,
                                 std::vector<float_t> clip_value)
{
    TransArray transArray(data, n_threads);
    transArray.computeNormalizedDifference(positive_indices, negative_indices, result_indices,
                                           positive_scaling, negative_scaling, result_scaling, result_offset, clip_value);
}


void computeNirv(Eigen::Ref<MatFloat> data,
                                 const uint_t n_threads,
                                 std::vector<uint_t> nir_indices,
                                 std::vector<uint_t> red_indices,
                                 std::vector<uint_t> result_indices,
                                 float_t nir_scaling,
                                 float_t red_scaling,
                                 float_t result_scaling,
                                 float_t result_offset,
                                 std::vector<float_t> clip_value)
{
    TransArray transArray(data, n_threads);
    transArray.computeNirv(nir_indices, red_indices, result_indices,
                          nir_scaling, red_scaling, result_scaling, result_offset, clip_value);
}

void computeBsi(Eigen::Ref<MatFloat> data,
                const uint_t n_threads,
                std::vector<uint_t> swir1_indices,
                std::vector<uint_t> red_indices,
                std::vector<uint_t> nir_indices,
                std::vector<uint_t> blue_indices,
                std::vector<uint_t> result_indices,
                float_t swir1_scaling,
                float_t red_scaling,
                float_t nir_scaling,
                float_t blue_scaling,
                float_t result_scaling,
                float_t result_offset,
                std::vector<float_t> clip_value)
{
    TransArray transArray(data, n_threads);
    transArray.computeBsi(swir1_indices, red_indices, nir_indices, blue_indices, result_indices,
                          swir1_scaling, red_scaling, nir_scaling, blue_scaling, result_scaling, result_offset, clip_value);

}


void computeEvi(Eigen::Ref<MatFloat> data,
                const uint_t n_threads,
                std::vector<uint_t> red_indices,
                std::vector<uint_t> nir_indices,
                std::vector<uint_t> blue_indices,
                std::vector<uint_t> result_indices,
                float_t red_scaling,
                float_t nir_scaling,
                float_t blue_scaling,
                float_t result_scaling,
                float_t result_offset,
                std::vector<float_t> clip_value)
{
    TransArray transArray(data, n_threads);
    transArray.computeEvi(red_indices, nir_indices, blue_indices, result_indices,
                          red_scaling, nir_scaling, blue_scaling, result_scaling, result_offset, clip_value);

}

void computeFapar(Eigen::Ref<MatFloat> data,
                const uint_t n_threads,
                std::vector<uint_t> red_indices,
                std::vector<uint_t> nir_indices,
                std::vector<uint_t> result_indices,
                float_t red_scaling,
                float_t nir_scaling,
                float_t result_scaling,
                float_t result_offset,
                std::vector<float_t> clip_value)
{
    TransArray transArray(data, n_threads);
    transArray.computeFapar(red_indices, nir_indices, result_indices,
                           red_scaling, nir_scaling, result_scaling, result_offset, clip_value);
}

void computeGeometricTemperature(Eigen::Ref<MatFloat> data,
                                 const uint_t n_threads,
                                 Eigen::Ref<MatFloat> latitude,
                                 Eigen::Ref<MatFloat> elevation,
                                 float_t elevation_scaling,
                                 float_t a,
                                 float_t b,
                                 float_t result_scaling,
                                 std::vector<uint_t> result_indices,
                                 std::vector<float_t> days_of_year)
{
    TransArray transArray(data, n_threads);
    transArray.computeGeometricTemperature(latitude, elevation, elevation_scaling, a, b, result_scaling, result_indices, days_of_year);
}

void writeByteData(Eigen::Ref<MatFloat> data,
                   const uint_t n_threads,
                   py::dict conf_GDAL,
                   std::vector<std::string> base_files,
                   std::string base_folder,
                   std::vector<std::string> file_names,
                   std::vector<uint_t> data_indices,
                   uint_t x_off,
                   uint_t y_off,
                   uint_t x_size,
                   uint_t y_size,
                   byte_t no_data_value,
                   std::optional<std::string> bash_compression_command,
                   std::optional<std::vector<std::string>> seaweed_path)
{
    IoArray ioArray(data, n_threads);
    ioArray.setupGdal(convPyDict(conf_GDAL));
    ioArray.writeData(base_files, base_folder, file_names, data_indices,
        x_off, y_off, x_size, y_size, GDT_Byte, no_data_value, bash_compression_command, seaweed_path);

}


void writeInt16Data(Eigen::Ref<MatFloat> data,
                   const uint_t n_threads,
                   py::dict conf_GDAL,
                   std::vector<std::string> base_files,
                   std::string base_folder,
                   std::vector<std::string> file_names,
                   std::vector<uint_t> data_indices,
                   uint_t x_off,
                   uint_t y_off,
                   uint_t x_size,
                   uint_t y_size,
                   int16_t no_data_value,
                   std::optional<std::string> bash_compression_command,
                   std::optional<std::vector<std::string>> seaweed_path)
{
    IoArray ioArray(data, n_threads);
    ioArray.setupGdal(convPyDict(conf_GDAL));
    ioArray.writeData(base_files, base_folder, file_names, data_indices,
        x_off, y_off, x_size, y_size, GDT_Int16, no_data_value, bash_compression_command, seaweed_path);

}


void warpTile(Eigen::Ref<MatFloat> data,
                    const uint_t n_threads,
                    py::dict conf_GDAL,
                    std::string tilePath,
                    std::string mosaicPath,
                    std::string resample)
{
    IoArray ioArray(data, n_threads);
    ioArray.setupGdal(convPyDict(conf_GDAL));
    ioArray.warpTile(tilePath, mosaicPath, resample);
}

void computePercentiles(Eigen::Ref<MatFloat> data,
                          const uint_t n_threads,
                          Eigen::Ref<MatFloat> out_data,
                          uint_t out_index_offset,
                          std::vector<float_t> percentiles)
{
    TransArray transArray(data, n_threads);
    transArray.computePercentiles(out_data, out_index_offset, percentiles);
}

void applySircle(Eigen::Ref<MatFloat> data,
                 const uint_t n_threads,
                 Eigen::Ref<MatFloat> out_data,
                 uint_t out_index_offset,
                 float_t w_0,
                 Eigen::Ref<VecFloat> w_p,
                 Eigen::Ref<VecFloat> w_f,
                 bool keep_original_values,
                 const std::string& version,
                 const std::string& backend)

{
    TransArray transArray(data, n_threads);
    transArray.applySircle(out_data, out_index_offset,
                           w_0, w_p, w_f, keep_original_values, version, backend);
}

PYBIND11_MODULE(skmap_bindings, m)
{
    m.def("readData", &readData,
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg() = std::nullopt, py::arg() = std::nullopt,
        "Read Tiff files in parallel with GDAL-Eigen-OpenMP");
    m.def("copyVecInMatrixRow", &copyVecInMatrixRow, "Copy a vector in a matrix row");
    m.def("fillArray", &fillArray, "Fill array");
    m.def("selArrayRows", &selArrayRows, "Mask array rows");
    m.def("averageAggregate", &averageAggregate, "Average aggregate");
    m.def("maskData", &maskData, "Mask data");
    m.def("maskNan", &maskNan, "Mask NaN");
    m.def("swapRowsValues", &swapRowsValues, "Swap array values");
    m.def("expandArrayRows", &expandArrayRows, "Expand array rows");
    m.def("extractArrayRows", &extractArrayRows, "Extract array rows");
    m.def("transposeArray", &transposeArray, "Transpose an array into a new one");
    m.def("reorderArray", &reorderArray, "Reorder an array into a new one");
    m.def("offsetAndScale", &offsetAndScale, "Add an offset and muplitply by a scaling each array element");
    m.def("scaleAndOffset", &scaleAndOffset, "Muplitply by a scaling and add an offset each array element");
    m.def("inverseReorderArray", &inverseReorderArray, "Reorder and transpose an array into a new one");
    m.def("writeByteData", &writeByteData, 
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg() = std::nullopt, py::arg() = std::nullopt,
        "Write data in Byte format");
    m.def("writeInt16Data", &writeInt16Data, 
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg() = std::nullopt, py::arg() = std::nullopt,
        "Write data in Int16 format");
    m.def("getLatLonArray", &getLatLonArray, "Compute latitude and longitude for each pixel of a GeoTIFF");
    m.def("computeNormalizedDifference", &computeNormalizedDifference, "Compute normalized difference indices");
    m.def("computeBsi", &computeBsi, "Compute BSI");
    m.def("computeEvi", &computeEvi, "Compute EVI");
    m.def("computeNirv", &computeNirv, "Compute NIRv");
    m.def("computeFapar", &computeFapar, "Compute FAPAR");
    m.def("nanMean", &nanMean, "Compute average between available values");
    m.def("computeMannKendallPValues", &computeMannKendallPValues, "Compute Mann-Kendall p-values");
    m.def("warpTile", &warpTile, "Compute FAPAR");
    m.def("linearRegression", &linearRegression, "Compute linear regression slope and intercept");
    m.def("transposeReorderArray", &transposeReorderArray, "Transpose and reorder an array into a new one");
    m.def("computeGeometricTemperature", &computeGeometricTemperature, "Compute geometric temperautre");
    m.def("computePercentiles", &computePercentiles, "Compute percentile");
    m.def("applySircle", &applySircle, "Apply SIRCLE");
    m.def("testLib", &testLib, "testLib");
    m.def("fitPercentage", &fitPercentage, "Fit a three percages to 100 starting from 2");
    m.def("hadamardProduct", &hadamardProduct, "Elemennt wise product");
    m.def("averageAi4sh", &averageAi4sh, "Werid vecorized average of 4 elemnts made ad-hoc for AI4SH production");
    m.def("maskDifference", &maskDifference, "Mask outliers by difference from a reference");
}

