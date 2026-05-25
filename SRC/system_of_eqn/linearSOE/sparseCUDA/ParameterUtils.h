#ifndef ParameterUtils_h
#define ParameterUtils_h

#include <unordered_map>
#include <string>
#include <functional>

namespace ParameterUtils {
    // Helper function to strip dashes and find parameter
    template<typename ConfigType>
    static auto findParameter(const std::string& key, 
                            const std::unordered_map<std::string, std::function<void(ConfigType&)>>& parsers) {
        // Strip leading dash if present
        if (!key.empty() && key[0] == '-') {
            return parsers.find(key.substr(1));
        }
        return parsers.find(key);
    }
}

// Process count for CuDSS parsing. With MPI, uses MPI_Comm_size; otherwise 1.
int getNumProcesses();

#endif // ParameterUtils_h
