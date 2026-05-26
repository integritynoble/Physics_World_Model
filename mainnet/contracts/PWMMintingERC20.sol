// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

interface IPWMRewardERC20 {
    function depositMinting(bytes32 benchmarkHash, uint256 amount) external;
}

/// @title PWMMintingERC20
/// @notice ERC20 (PWM token) sibling of PWMMinting.
///
/// The minting pool is pre-funded by transferring 17,220,000 PWM into this
/// contract at genesis. mintFor() emits A_{k,j,b} per the Zeno formula and
/// forwards it to PWMRewardERC20.depositMinting via approve + transferFrom.
///
/// Weight formulas (unchanged from PWMMinting):
///   w_k       = δ_k × max(activity_k, 1)
///   w_{k,j,b} = ρ_{j,b} × max(activity_{k,j,b}, ρ_{j,b})
///   A_k       = remaining × w_k / Σ(w_j)
///   A_{k,j,b} = A_k × w_{k,j,b} / Σ(w_{k,j',b'})
contract PWMMintingERC20 {
    using SafeERC20 for IERC20;

    uint256 public constant M_POOL = 17_220_000 ether;

    IERC20  public immutable pwmToken;
    address public governance;
    address public certificate;
    IPWMRewardERC20 public reward;

    uint256 public M_emitted;

    // Soft-launch posture: minting is paused at deploy; governance unpauses
    // when grant-funded audit clears and caps are raised. Per-event A_{k,j,b}
    // mint flows from PWMCertificate.finalize → mintFor are blocked while paused.
    bool public mintingPaused;

    struct Principle {
        bool      promoted;
        uint256   delta;
        uint256   activity;
        bytes32[] benchmarks;
    }
    mapping(uint256 => Principle) private _principles;

    struct Benchmark {
        bool    registered;
        uint256 rho;
        uint256 activity;
        uint256 index;
    }
    mapping(uint256 => mapping(bytes32 => Benchmark)) private _benchmarks;

    uint256 public totalPrincipleWeight;
    mapping(uint256 => uint256) public sumBenchmarkWeight;

    event GovernanceUpdated(address indexed newGovernance);
    event CertificateUpdated(address indexed newCertificate);
    event RewardUpdated(address indexed newReward);
    event MintingPausedUpdated(bool paused);

    event PromotionSet(uint256 indexed principleId, bool promoted);
    event DeltaSet(uint256 indexed principleId, uint256 delta);
    event BenchmarkRegistered(uint256 indexed principleId, bytes32 indexed benchmarkHash, uint256 rho);
    event BenchmarkRhoUpdated(uint256 indexed principleId, bytes32 indexed benchmarkHash, uint256 rho);
    event BenchmarkRemoved(uint256 indexed principleId, bytes32 indexed benchmarkHash);
    event ActivityRecorded(
        uint256 indexed principleId,
        bytes32 indexed benchmarkHash,
        uint256 principleActivity,
        uint256 benchmarkActivity
    );
    event Minted(
        uint256 indexed principleId,
        bytes32 indexed benchmarkHash,
        uint256 A_k,
        uint256 A_kjb,
        uint256 remainingAfter
    );

    modifier onlyGovernance()  { require(msg.sender == governance,  "PWMMintingERC20: not governance");  _; }
    modifier onlyCertificate() { require(msg.sender == certificate, "PWMMintingERC20: not certificate"); _; }

    constructor(address token_, address initialGovernance) {
        require(token_ != address(0), "PWMMintingERC20: zero token");
        require(initialGovernance != address(0), "PWMMintingERC20: zero governance");
        pwmToken   = IERC20(token_);
        governance = initialGovernance;
    }

    // ---------- wiring ----------

    function setGovernance(address x) external onlyGovernance {
        require(x != address(0), "PWMMintingERC20: zero governance");
        governance = x; emit GovernanceUpdated(x);
    }
    function setCertificate(address x) external onlyGovernance {
        require(x != address(0), "PWMMintingERC20: zero certificate");
        certificate = x; emit CertificateUpdated(x);
    }
    function setReward(address x) external onlyGovernance {
        require(x != address(0), "PWMMintingERC20: zero reward");
        reward = IPWMRewardERC20(x); emit RewardUpdated(x);
    }

    function setMintingPaused(bool paused) external onlyGovernance {
        mintingPaused = paused;
        emit MintingPausedUpdated(paused);
    }

    // ---------- principle management ----------

    function setDelta(uint256 principleId, uint256 delta) external onlyGovernance {
        require(delta > 0, "PWMMintingERC20: zero delta");
        Principle storage p = _principles[principleId];
        uint256 oldW = _principleWeight(principleId);
        p.delta = delta;
        uint256 newW = _principleWeight(principleId);
        if (p.promoted) {
            totalPrincipleWeight = totalPrincipleWeight - oldW + newW;
        }
        emit DeltaSet(principleId, delta);
    }

    function setPromotion(uint256 principleId, bool status) external onlyGovernance {
        Principle storage p = _principles[principleId];
        if (status && !p.promoted) {
            require(p.delta > 0, "PWMMintingERC20: delta unset");
            require(p.benchmarks.length > 0, "PWMMintingERC20: no benchmarks");
            p.promoted = true;
            totalPrincipleWeight += _principleWeight(principleId);
        } else if (!status && p.promoted) {
            totalPrincipleWeight -= _principleWeight(principleId);
            p.promoted = false;
        }
        emit PromotionSet(principleId, status);
    }

    function registerBenchmark(uint256 principleId, bytes32 benchmarkHash, uint256 rho) external onlyGovernance {
        require(benchmarkHash != bytes32(0), "PWMMintingERC20: zero benchmark");
        require(rho > 0, "PWMMintingERC20: zero rho");
        Principle storage p = _principles[principleId];
        Benchmark storage b = _benchmarks[principleId][benchmarkHash];
        require(!b.registered, "PWMMintingERC20: already registered");

        b.registered = true;
        b.rho = rho;
        b.index = p.benchmarks.length;
        p.benchmarks.push(benchmarkHash);

        sumBenchmarkWeight[principleId] += _benchmarkWeight(principleId, benchmarkHash);
        emit BenchmarkRegistered(principleId, benchmarkHash, rho);
    }

    function setBenchmarkRho(uint256 principleId, bytes32 benchmarkHash, uint256 rho) external onlyGovernance {
        require(rho > 0, "PWMMintingERC20: zero rho");
        Benchmark storage b = _benchmarks[principleId][benchmarkHash];
        require(b.registered, "PWMMintingERC20: unknown benchmark");
        uint256 oldW = _benchmarkWeight(principleId, benchmarkHash);
        b.rho = rho;
        uint256 newW = _benchmarkWeight(principleId, benchmarkHash);
        sumBenchmarkWeight[principleId] = sumBenchmarkWeight[principleId] - oldW + newW;
        emit BenchmarkRhoUpdated(principleId, benchmarkHash, rho);
    }

    function removeBenchmark(uint256 principleId, bytes32 benchmarkHash) external onlyGovernance {
        Principle storage p = _principles[principleId];
        Benchmark storage b = _benchmarks[principleId][benchmarkHash];
        require(b.registered, "PWMMintingERC20: unknown benchmark");
        if (p.promoted) {
            require(p.benchmarks.length > 1, "PWMMintingERC20: cannot leave zero benchmarks");
        }
        sumBenchmarkWeight[principleId] -= _benchmarkWeight(principleId, benchmarkHash);

        uint256 idx = b.index;
        uint256 last = p.benchmarks.length - 1;
        if (idx != last) {
            bytes32 moved = p.benchmarks[last];
            p.benchmarks[idx] = moved;
            _benchmarks[principleId][moved].index = idx;
        }
        p.benchmarks.pop();
        delete _benchmarks[principleId][benchmarkHash];

        emit BenchmarkRemoved(principleId, benchmarkHash);
    }

    // ---------- per-event mint ----------

    function mintFor(uint256 principleId, bytes32 benchmarkHash)
        external
        onlyCertificate
        returns (uint256 A_kjb)
    {
        require(!mintingPaused, "PWMMintingERC20: minting paused");
        Principle storage p = _principles[principleId];
        Benchmark storage b = _benchmarks[principleId][benchmarkHash];
        require(p.promoted, "PWMMintingERC20: not promoted");
        require(b.registered, "PWMMintingERC20: unknown benchmark");
        require(address(reward) != address(0), "PWMMintingERC20: reward unset");

        uint256 rem = remaining();
        uint256 sumW = totalPrincipleWeight;
        uint256 wK   = _principleWeight(principleId);

        uint256 A_k = 0;
        if (rem > 0 && sumW > 0 && wK > 0) {
            A_k = (rem * wK) / sumW;
        }

        uint256 sumBW = sumBenchmarkWeight[principleId];
        uint256 wB    = _benchmarkWeight(principleId, benchmarkHash);
        if (A_k > 0 && sumBW > 0 && wB > 0) {
            A_kjb = (A_k * wB) / sumBW;
        }

        if (A_kjb > 0) {
            require(pwmToken.balanceOf(address(this)) >= A_kjb, "PWMMintingERC20: underfunded");
            M_emitted += A_kjb;
        }

        // CEI: update state before external call.
        _incrementActivity(principleId, benchmarkHash);

        emit Minted(principleId, benchmarkHash, A_k, A_kjb, M_POOL - M_emitted);

        if (A_kjb > 0) {
            pwmToken.forceApprove(address(reward), A_kjb);
            reward.depositMinting(benchmarkHash, A_kjb);
        }
    }

    function _incrementActivity(uint256 principleId, bytes32 benchmarkHash) internal {
        Principle storage p = _principles[principleId];
        Benchmark storage b = _benchmarks[principleId][benchmarkHash];

        uint256 oldBW = _benchmarkWeight(principleId, benchmarkHash);
        uint256 oldPW = _principleWeight(principleId);

        p.activity += 1;
        b.activity += 1;

        uint256 newBW = _benchmarkWeight(principleId, benchmarkHash);
        sumBenchmarkWeight[principleId] = sumBenchmarkWeight[principleId] - oldBW + newBW;

        if (p.promoted) {
            uint256 newPW = _principleWeight(principleId);
            totalPrincipleWeight = totalPrincipleWeight - oldPW + newPW;
        }

        emit ActivityRecorded(principleId, benchmarkHash, p.activity, b.activity);
    }

    // ---------- views ----------

    function remaining() public view returns (uint256) {
        return M_POOL - M_emitted;
    }

    function principleOf(uint256 principleId)
        external view
        returns (bool promoted, uint256 delta, uint256 activity, uint256 numBenchmarks)
    {
        Principle storage p = _principles[principleId];
        return (p.promoted, p.delta, p.activity, p.benchmarks.length);
    }

    function benchmarkOf(uint256 principleId, bytes32 benchmarkHash)
        external view
        returns (bool registered, uint256 rho, uint256 activity)
    {
        Benchmark storage b = _benchmarks[principleId][benchmarkHash];
        return (b.registered, b.rho, b.activity);
    }

    function principleWeight(uint256 principleId) external view returns (uint256) {
        if (!_principles[principleId].promoted) return 0;
        return _principleWeight(principleId);
    }

    function benchmarkWeight(uint256 principleId, bytes32 benchmarkHash) external view returns (uint256) {
        if (!_benchmarks[principleId][benchmarkHash].registered) return 0;
        return _benchmarkWeight(principleId, benchmarkHash);
    }

    function _principleWeight(uint256 principleId) internal view returns (uint256) {
        Principle storage p = _principles[principleId];
        uint256 a = p.activity == 0 ? 1 : p.activity;
        return p.delta * a;
    }

    function _benchmarkWeight(uint256 principleId, bytes32 benchmarkHash) internal view returns (uint256) {
        Benchmark storage b = _benchmarks[principleId][benchmarkHash];
        uint256 a = b.activity < b.rho ? b.rho : b.activity;
        return b.rho * a;
    }
}
