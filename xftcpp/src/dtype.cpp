#include "xftcpp/src/dtype.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/xla_data.pb.h"

namespace xftcpp {

// ============================================================================
// Utility Methods: byte_size()
// ============================================================================

std::optional<int> DType::byte_size() const {
  switch (kind_) {
    
    // Sub-byte types - cannot be represented as whole bytes
    case kS2:
    case kU2:
    case kS4:
    case kU4:
    case kF4E2M1FN:
      return std::nullopt;
    
    // 1-byte types
    case kPred:
    case kS8:
    case kU8:
    case kF8E3M4:
    case kF8E4M3:
    case kF8E8M0FNU:
    case kF8E4M3FN:
    case kF8E4M3B11FNUZ:
    case kF8E4M3FNUZ:
    case kF8E5M2:
    case kF8E5M2FNUZ:
      return 1;
    
    // 2-byte types
    case kS16:
    case kU16:
    case kF16:
    case kBF16:
      return 2;
    
    // 4-byte types
    case kS32:
    case kU32:
    case kF32:
      return 4;

    // 8-byte types
    case kS64:
    case kU64:
    case kF64:
    case kC64:
      return 8;
    
    // 16-byte types
    case kC128:
      return 16;
    
    // Types with no fixed size
    case kToken:
    case kOpaque:
    case kInvalid:
    case kString:
      return std::nullopt;
  }

   // Should never reach here, but return nullopt for safety
  return std::nullopt;
}


// ============================================================================
// Utility Methods: bit_size()
// ============================================================================

std::optional<int> DType::bit_size() const {
  switch (kind_) {
    // 2-bit types
    case kS2:
    case kU2:
      return 2;

    // 4-bit types
    case kS4:
    case kU4:
    case kF4E2M1FN:
      return 4;
    
     // 8-bit types
    case kPred:
    case kS8:
    case kU8:
    case kF8E3M4:
    case kF8E4M3:
    case kF8E8M0FNU:
    case kF8E4M3FN:
    case kF8E4M3B11FNUZ:
    case kF8E4M3FNUZ:
    case kF8E5M2:
    case kF8E5M2FNUZ:
      return 8;
    
    // 16-bit types
    case kS16:
    case kU16:
    case kF16:
    case kBF16:
      return 16;

    // 32-bit types
    case kS32:
    case kU32:
    case kF32:
      return 32;

    // 64-bit types
    case kS64:
    case kU64:
    case kF64:
    case kC64:
      return 64;
    
    // 128-bit types
    case kC128:
      return 128;

    // Types with no fixed size
    case kToken:
    case kOpaque:
    case kInvalid:
    case kString:
      return std::nullopt;
  }

  // Should never reach here, but return nullopt for safety
  return std::nullopt;
}

// ============================================================================
// Utility Methods: DebugString()
// ============================================================================

std::string DType::DebugString() const {
  switch (kind_) {
    case kInvalid:
      return "INVALID";
    case kPred:
      return "PRED";
    case kS2:
      return "S2";
    case kS4:
      return "S4";
    case kS8:
      return "S8";
    case kS16:
      return "S16";
    case kS32:
      return "S32";
    case kS64:
      return "S64";
    case kU2:
      return "U2";
    case kU4:
      return "U4";
    case kU8:
      return "U8";
    case kU16:
      return "U16";
    case kU32:
      return "U32";
    case kU64:
      return "U64";
    case kF16:
      return "F16";
    case kF32:
      return "F32";
    case kF64:
      return "F64";
    case kBF16:
      return "BF16";
    case kC64:
      return "C64";
    case kC128:
      return "C128";
    case kToken:
      return "TOKEN";
    case kOpaque:
      return "OPAQUE";
    case kF4E2M1FN:
      return "F4E2M1FN";
    case kF8E3M4:
      return "F8E3M4";
    case kF8E4M3:
      return "F8E4M3";
    case kF8E4M3FN:
      return "F8E4M3FN";
    case kF8E4M3B11FNUZ:
      return "F8E4M3B11FNUZ";
    case kF8E4M3FNUZ:
      return "F8E4M3FNUZ";
    case kF8E5M2:
      return "F8E5M2";
    case kF8E5M2FNUZ:
      return "F8E5M2FNUZ";
    case kF8E8M0FNU:
      return "F8E8M0FNU";
    case kString:
      return "STRING";
    default:
      return absl::StrCat("UNKNOWN(", static_cast<int>(kind_), ")");
  }
}

// ============================================================================
// Stream Output Operator
// ============================================================================

std::ostream& operator<<(std::ostream& os, const DType& dtype) {
  return os << dtype.DebugString();
}

// ============================================================================
// XLA Conversion: DType -> xla::PrimitiveType
// ============================================================================

absl::StatusOr<xla::PrimitiveType> ToPrimitiveType(DType dtype) {
  switch (dtype.kind()) {
    // Use macro for type conversion with compile-time validation
    // This ensures DType::Kind enum values match xla::PrimitiveType values
#define CASE(DT, PT)                                                      \
  case DT:                                                                \
    static_assert(PT ==                                                   \
                  static_cast<xla::PrimitiveType>(static_cast<int>(DT))); \
    return PT
    CASE(DType::kInvalid, xla::PrimitiveType::PRIMITIVE_TYPE_INVALID);
    CASE(DType::kPred, xla::PrimitiveType::PRED);
    CASE(DType::kS2, xla::PrimitiveType::S2);
    CASE(DType::kS4, xla::PrimitiveType::S4);
    CASE(DType::kS8, xla::PrimitiveType::S8);
    CASE(DType::kS16, xla::PrimitiveType::S16);
    CASE(DType::kS32, xla::PrimitiveType::S32);
    CASE(DType::kS64, xla::PrimitiveType::S64);
    CASE(DType::kU2, xla::PrimitiveType::U2);
    CASE(DType::kU4, xla::PrimitiveType::U4);
    CASE(DType::kU8, xla::PrimitiveType::U8);
    CASE(DType::kU16, xla::PrimitiveType::U16);
    CASE(DType::kU32, xla::PrimitiveType::U32);
    CASE(DType::kU64, xla::PrimitiveType::U64);
    CASE(DType::kF4E2M1FN, xla::PrimitiveType::F4E2M1FN);
    CASE(DType::kF8E3M4, xla::PrimitiveType::F8E3M4);
    CASE(DType::kF8E4M3, xla::PrimitiveType::F8E4M3);
    CASE(DType::kF8E4M3FN, xla::PrimitiveType::F8E4M3FN);
    CASE(DType::kF8E4M3B11FNUZ, xla::PrimitiveType::F8E4M3B11FNUZ);
    CASE(DType::kF8E4M3FNUZ, xla::PrimitiveType::F8E4M3FNUZ);
    CASE(DType::kF8E5M2, xla::PrimitiveType::F8E5M2);
    CASE(DType::kF8E5M2FNUZ, xla::PrimitiveType::F8E5M2FNUZ);
    CASE(DType::kF8E8M0FNU, xla::PrimitiveType::F8E8M0FNU);
    CASE(DType::kF16, xla::PrimitiveType::F16);
    CASE(DType::kF32, xla::PrimitiveType::F32);
    CASE(DType::kBF16, xla::PrimitiveType::BF16);
    CASE(DType::kF64, xla::PrimitiveType::F64);
    CASE(DType::kC64, xla::PrimitiveType::C64);
    CASE(DType::kC128, xla::PrimitiveType::C128);
    CASE(DType::kToken, xla::PrimitiveType::TOKEN);
    CASE(DType::kOpaque, xla::PrimitiveType::OPAQUE_TYPE);

#undef CASE

    // kString is not supported by XLA PJRT
    case DType::kString:
      return absl::InvalidArgumentError(
          absl::StrCat("Not supported as XLA PrimitiveType: ",
                       static_cast<int>(dtype.kind())));
  }

  // Should never reach here if all enum values are handled
  return absl::InvalidArgumentError(
      absl::StrCat("Invalid DType: ", static_cast<int>(dtype.kind())));
}

// ============================================================================
// XLA Conversion: xla::PrimitiveType -> DType
// ============================================================================

absl::StatusOr<DType> FromPrimitiveType(xla::PrimitiveType primitive_type) {
  // Since DType::Kind values match xla::PrimitiveType exactly (validated
  // by static_assert in ToPrimitiveType), we can directly cast for all
  // supported types.
   switch (primitive_type) {
    case xla::PrimitiveType::PRIMITIVE_TYPE_INVALID:
    case xla::PrimitiveType::PRED:
    case xla::PrimitiveType::S2:
    case xla::PrimitiveType::S4:
    case xla::PrimitiveType::S8:
    case xla::PrimitiveType::S16:
    case xla::PrimitiveType::S32:
    case xla::PrimitiveType::S64:
    case xla::PrimitiveType::U2:
    case xla::PrimitiveType::U4:
    case xla::PrimitiveType::U8:
    case xla::PrimitiveType::U16:
    case xla::PrimitiveType::U32:
    case xla::PrimitiveType::U64:
    case xla::PrimitiveType::F4E2M1FN:
    case xla::PrimitiveType::F8E3M4:
    case xla::PrimitiveType::F8E4M3:
    case xla::PrimitiveType::F8E4M3FN:
    case xla::PrimitiveType::F8E4M3B11FNUZ:
    case xla::PrimitiveType::F8E4M3FNUZ:
    case xla::PrimitiveType::F8E5M2:
    case xla::PrimitiveType::F8E5M2FNUZ:
    case xla::PrimitiveType::F8E8M0FNU:
    case xla::PrimitiveType::F16:
    case xla::PrimitiveType::F32:
    case xla::PrimitiveType::BF16:
    case xla::PrimitiveType::F64:
    case xla::PrimitiveType::C64:
    case xla::PrimitiveType::C128:
    case xla::PrimitiveType::TOKEN:
    case xla::PrimitiveType::OPAQUE_TYPE:
      // Direct cast is safe because enum values match
      return DType(static_cast<DType::Kind>(static_cast<int>(primitive_type)));
    default:
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid or unsupported XLA PrimitiveType: ",
                        static_cast<int>(primitive_type)));
  }
}

}   // namespace xftcpp