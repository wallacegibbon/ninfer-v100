#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::artifact {

// Container framing revisions share the first eight bytes' shape: ASCII "NINFER", one zero byte and
// the revision. Continuation volumes of a v3 file set use a different magic so a stray part cannot
// be mistaken for an entry.
inline constexpr std::array<std::byte, 8> kEntryMagicV2 = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{2},
};
inline constexpr std::array<std::byte, 8> kEntryMagicV3 = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{3},
};
inline constexpr std::array<std::byte, 8> kPartMagicV3 = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'P'},
    std::byte{'R'}, std::byte{'T'}, std::byte{0},   std::byte{3},
};

inline constexpr std::uint64_t kPayloadAlignment = 4096;
inline constexpr std::size_t kHeaderV2Bytes      = 16;
inline constexpr std::size_t kHeaderV3Bytes      = 32;
inline constexpr std::size_t kArtifactIdBytes    = 16;

inline std::uint64_t read_u64_le(const std::byte* bytes) noexcept {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= std::uint64_t(std::to_integer<unsigned char>(bytes[i])) << (i * 8);
    }
    return value;
}

} // namespace ninfer::artifact
