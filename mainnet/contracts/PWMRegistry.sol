// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/access/Ownable.sol";

/// @title PWMRegistry
/// @notice Immutable artifact hash store for L1/L2/L3/L4 artifacts.
///         Append-only: no delete, no update. Every PWM protocol reference
///         resolves to an entry here.
///         register() is restricted to the contract owner (Director's deployer EOA).
///         Use transferOwnership() to rotate to a new deployer key if needed.
contract PWMRegistry is Ownable {
    struct Artifact {
        bytes32 parentHash;
        uint8   layer;        // 1=Principle, 2=Spec, 3=Benchmark, 4=Solution
        address creator;
        uint256 timestamp;
    }

    mapping(bytes32 => Artifact) private _artifacts;

    event ArtifactRegistered(
        bytes32 indexed hash,
        uint8           layer,
        address indexed creator,
        uint256         timestamp
    );

    constructor() Ownable(msg.sender) {}

    /// @notice Register a new artifact. Reverts if hash already exists or layer invalid.
    ///         Only the contract owner (Director's deployer EOA) may call this.
    function register(
        bytes32 hash,
        bytes32 parentHash,
        uint8   layer,
        address creator
    ) external onlyOwner {
        require(hash != bytes32(0), "PWMRegistry: zero hash");
        require(_artifacts[hash].timestamp == 0, "PWMRegistry: already registered");
        require(layer >= 1 && layer <= 4, "PWMRegistry: invalid layer");
        require(creator != address(0), "PWMRegistry: zero creator");
        // L2/L3/L4 must reference a parent. L1 (Principle) is a root.
        if (layer == 1) {
            require(parentHash == bytes32(0), "PWMRegistry: L1 must have zero parent");
        } else {
            require(parentHash != bytes32(0), "PWMRegistry: parent required");
            require(_artifacts[parentHash].timestamp != 0, "PWMRegistry: parent not registered");
        }

        _artifacts[hash] = Artifact({
            parentHash: parentHash,
            layer:      layer,
            creator:    creator,
            timestamp:  block.timestamp
        });

        emit ArtifactRegistered(hash, layer, creator, block.timestamp);
    }

    /// @notice Retrieve artifact metadata. Returns zero values if not registered.
    function getArtifact(bytes32 hash)
        external
        view
        returns (bytes32 parentHash, uint8 layer, address creator, uint256 timestamp)
    {
        Artifact storage a = _artifacts[hash];
        return (a.parentHash, a.layer, a.creator, a.timestamp);
    }

    /// @notice Returns true iff hash is registered.
    function exists(bytes32 hash) external view returns (bool) {
        return _artifacts[hash].timestamp != 0;
    }
}
