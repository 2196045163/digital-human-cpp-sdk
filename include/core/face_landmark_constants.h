#pragma once

#include <cstddef>

namespace digital_human {
namespace core {

inline constexpr std::size_t kFaceLandmarkCount = 68;

inline constexpr int kMouthLandmarkStart = 48;
inline constexpr int kMouthOuterLandmarkEnd = 59;
inline constexpr int kMouthFullLandmarkEnd = 67;
inline constexpr int kMouthFullLandmarkEndExclusive = kMouthFullLandmarkEnd + 1;

} // namespace core
} // namespace digital_human

