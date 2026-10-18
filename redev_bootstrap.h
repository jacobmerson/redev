#ifndef REDEV_REDEV_BOOTSTRAP_H
#define REDEV_REDEV_BOOTSTRAP_H
#include "redev_config.h"
#include <map>
#include <mpi.h>
#include <string>

namespace redev {

/**
 * Data the server sends a client before their channel is opened, i.e. only
 * what is needed to open it (currently the transport). Everything else should
 * be exchanged over the open channel.
 *
 * It is written by server rank 0 as a small versioned text file next to the
 * channel's other files and is independent of any backend:
 * \code
 * redev-bootstrap 1
 * transport=bp4
 * \endcode
 * Readers ignore keys they do not know, so new keys can be added without
 * changing the version. The client removes the file after reading it.
 */
using BootstrapData = std::map<std::string, std::string>;

/** How long a client waits for the bootstrap file, and how often it checks. */
struct BootstrapSettings {
  double timeoutSecs = 60;
  double pollSecs = 0.05;
};

/** Format version written by SendBootstrap. */
constexpr int kBootstrapVersion = 1;

/** File used for the bootstrap of the channel with this path + name. */
[[nodiscard]] std::string BootstrapFileName(const std::string& section);

/**
 * BootstrapSettings from the "redev" group of the channel's section
 * (BootstrapTimeoutSecs, BootstrapPollSecs). Invalid values are fatal.
 */
[[nodiscard]] BootstrapSettings GetBootstrapSettings(const Config& config,
                                                     const std::string& section);

/**
 * Write the bootstrap for a channel. Rank 0 of comm writes the file
 * atomically (write, then rename). Keys must be non-empty and contain no '='
 * or newline; values must contain no newline.
 */
void SendBootstrap(const std::string& section, const BootstrapData& data,
                   MPI_Comm comm);

/**
 * Wait for, read, and remove the bootstrap for a channel. Rank 0 of comm reads
 * the file and broadcasts its contents. Collective over comm. A timeout, a
 * newer format version, or a malformed file is fatal on every rank.
 */
[[nodiscard]] BootstrapData ReceiveBootstrap(const std::string& section,
                                             const BootstrapSettings& settings,
                                             MPI_Comm comm);

} // namespace redev
#endif
