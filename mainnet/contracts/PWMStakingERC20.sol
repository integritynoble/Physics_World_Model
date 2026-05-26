// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

interface IPWMRewardERC20 {
    function seedBPool(bytes32 benchmarkHash, uint256 amount) external;
}

/// @title PWMStakingERC20
/// @notice ERC20 (PWM token) sibling of PWMStaking. Same fates, same governance
///         model, same caps — but value is moved via IERC20 calls instead of
///         native coin transfers.
///
/// Tier defaults (governance-tunable):
///   L1 Principle:    10 PWM
///   L2 Spec:          2 PWM
///   L3 I-benchmark:   1 PWM
///
/// Stake fates (unchanged from native version):
///   Graduation: 50% returned to staker, 50% → PWMRewardERC20.seedBPool
///   Challenge upheld: 50% burned, 50% to challenger
///   Fraud: 100% burned
contract PWMStakingERC20 {
    using SafeERC20 for IERC20;

    uint8  public constant LAYER_PRINCIPLE = 1;
    uint8  public constant LAYER_SPEC      = 2;
    uint8  public constant LAYER_BENCHMARK = 3;

    address public constant BURN_SINK = 0x000000000000000000000000000000000000dEaD;

    IERC20  public immutable pwmToken;
    address public governance;
    IPWMRewardERC20 public reward;

    mapping(uint8 => uint256) public stakeAmount; // layer => required PWM amount (wei-precision)

    uint256 public totalActiveStakeWei;
    uint256 public maxTotalStakeWei;

    enum Status { None, Active, Graduated, Slashed, Fraud }
    struct Stake {
        address staker;
        uint8   layer;
        uint256 amount;
        Status  status;
    }
    mapping(bytes32 => Stake) public stakes;

    event StakeAmountUpdated(uint8 indexed layer, uint256 amount);
    event Staked(bytes32 indexed artifactHash, uint8 indexed layer, address indexed staker, uint256 amount);
    event Graduated(bytes32 indexed artifactHash, address indexed staker, uint256 returned, uint256 seeded);
    event ChallengeUpheld(bytes32 indexed artifactHash, address indexed challenger, uint256 burned, uint256 toChallenger);
    event FraudSlashed(bytes32 indexed artifactHash, uint256 burned);
    event GovernanceUpdated(address indexed newGovernance);
    event RewardUpdated(address indexed newReward);
    event MaxTotalStakeWeiUpdated(uint256 newMax);

    modifier onlyGovernance() { require(msg.sender == governance, "PWMStakingERC20: not governance"); _; }

    constructor(address token_, address initialGovernance) {
        require(token_ != address(0), "PWMStakingERC20: zero token");
        require(initialGovernance != address(0), "PWMStakingERC20: zero governance");
        pwmToken   = IERC20(token_);
        governance = initialGovernance;
        stakeAmount[LAYER_PRINCIPLE] = 10 ether;
        stakeAmount[LAYER_SPEC]      = 2 ether;
        stakeAmount[LAYER_BENCHMARK] = 1 ether;
    }

    // ---------- governance ----------

    function setGovernance(address x) external onlyGovernance {
        require(x != address(0), "PWMStakingERC20: zero governance");
        governance = x;
        emit GovernanceUpdated(x);
    }
    function setReward(address x) external onlyGovernance {
        require(x != address(0), "PWMStakingERC20: zero reward");
        reward = IPWMRewardERC20(x);
        emit RewardUpdated(x);
    }
    function setStakeAmount(uint8 layer, uint256 amount) external onlyGovernance {
        require(layer >= LAYER_PRINCIPLE && layer <= LAYER_BENCHMARK, "PWMStakingERC20: bad layer");
        require(amount > 0, "PWMStakingERC20: zero amount");
        stakeAmount[layer] = amount;
        emit StakeAmountUpdated(layer, amount);
    }
    function setMaxTotalStakeWei(uint256 newMax) external onlyGovernance {
        require(newMax > 0, "PWMStakingERC20: zero max disables cap; use governance proposal explicitly");
        maxTotalStakeWei = newMax;
        emit MaxTotalStakeWeiUpdated(newMax);
    }

    // ---------- staking ----------

    /// @notice Stake the exact per-layer amount in PWM tokens against an artifact.
    ///         Caller must have approved this contract for at least `stakeAmount[layer]` PWM.
    function stake(uint8 layer, bytes32 artifactHash) external {
        require(layer >= LAYER_PRINCIPLE && layer <= LAYER_BENCHMARK, "PWMStakingERC20: bad layer");
        require(artifactHash != bytes32(0), "PWMStakingERC20: zero hash");
        require(stakes[artifactHash].status == Status.None, "PWMStakingERC20: already staked");
        uint256 required = stakeAmount[layer];

        uint256 newTotal = totalActiveStakeWei + required;
        uint256 cap = maxTotalStakeWei;
        if (cap != 0) {
            require(newTotal <= cap, "PWMStakingERC20: total stake cap exceeded");
        }
        totalActiveStakeWei = newTotal;

        stakes[artifactHash] = Stake({
            staker: msg.sender,
            layer:  layer,
            amount: required,
            status: Status.Active
        });

        pwmToken.safeTransferFrom(msg.sender, address(this), required);
        emit Staked(artifactHash, layer, msg.sender, required);
    }

    // ---------- resolution ----------

    function graduate(bytes32 artifactHash, bytes32 benchmarkHash) external onlyGovernance {
        Stake storage s = stakes[artifactHash];
        require(s.status == Status.Active, "PWMStakingERC20: not active");
        require(address(reward) != address(0), "PWMStakingERC20: reward unset");
        require(benchmarkHash != bytes32(0), "PWMStakingERC20: zero benchmark");

        s.status = Status.Graduated;
        totalActiveStakeWei -= s.amount;
        uint256 half = s.amount / 2;
        uint256 other = s.amount - half;

        pwmToken.safeTransfer(s.staker, half);
        pwmToken.forceApprove(address(reward), other);
        reward.seedBPool(benchmarkHash, other);

        emit Graduated(artifactHash, s.staker, half, other);
    }

    function slashForChallenge(bytes32 artifactHash, address challenger) external onlyGovernance {
        Stake storage s = stakes[artifactHash];
        require(s.status == Status.Active, "PWMStakingERC20: not active");
        require(challenger != address(0), "PWMStakingERC20: zero challenger");

        s.status = Status.Slashed;
        totalActiveStakeWei -= s.amount;
        uint256 half = s.amount / 2;
        uint256 other = s.amount - half;

        pwmToken.safeTransfer(BURN_SINK, half);
        pwmToken.safeTransfer(challenger, other);

        emit ChallengeUpheld(artifactHash, challenger, half, other);
    }

    function slashForFraud(bytes32 artifactHash) external onlyGovernance {
        Stake storage s = stakes[artifactHash];
        require(s.status == Status.Active, "PWMStakingERC20: not active");

        s.status = Status.Fraud;
        totalActiveStakeWei -= s.amount;
        uint256 amt = s.amount;

        pwmToken.safeTransfer(BURN_SINK, amt);

        emit FraudSlashed(artifactHash, amt);
    }

    // ---------- views ----------

    function stakeOf(bytes32 artifactHash)
        external
        view
        returns (address staker, uint8 layer, uint256 amount, Status status)
    {
        Stake storage s = stakes[artifactHash];
        return (s.staker, s.layer, s.amount, s.status);
    }
}
