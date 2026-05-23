// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

interface IPWMTreasuryERC20 {
    function receive15pct(uint256 principleId, uint256 amount) external;
}

/// @title PWMRewardERC20
/// @notice ERC20 (PWM token) sibling of PWMReward.
///
/// Caller flow for deposits (seedBPool / depositMinting / depositBounty):
///   1. caller approves this contract for `amount` PWM
///   2. caller calls the deposit function with `amount`
///   3. this contract pulls via transferFrom and credits the benchmark pool
///
/// Splits and ranked-draw schedule are identical to PWMReward.
contract PWMRewardERC20 {
    using SafeERC20 for IERC20;

    uint16 public constant BPS_DENOM      = 10_000;
    uint16 public constant SPLIT_AC_CP    = 5_500;
    uint16 public constant SPLIT_L3       = 1_500;
    uint16 public constant SPLIT_L2       = 1_000;
    uint16 public constant SPLIT_L1       =   500;
    uint16 public constant SPLIT_TREASURY = 1_500;
    uint8  public constant MAX_RANK       = 10;

    IERC20  public immutable pwmToken;
    address public governance;
    address public certificate;
    address public minting;
    address public staking;
    IPWMTreasuryERC20 public treasury;

    mapping(bytes32 => uint256) public pool;
    mapping(bytes32 => bool)    public settled;

    uint256 public maxBenchmarkPoolWei;

    struct Draw {
        bytes32 benchmarkHash;
        uint256 principleId;
        address l1Creator;
        address l2Creator;
        address l3Creator;
        address acWallet;
        address cpWallet;
        uint16  shareRatioP;
        uint8   rank;
    }

    event PoolSeeded(bytes32 indexed benchmarkHash, uint256 amount, uint256 newBalance, address indexed from, string kind);
    event DrawSettled(
        bytes32 indexed certHash,
        bytes32 indexed benchmarkHash,
        uint8           rank,
        uint256         drawAmount,
        uint256         rolloverRemaining
    );
    event RoyaltiesPaid(bytes32 indexed certHash, address ac, uint256 acAmt, address cp, uint256 cpAmt, uint256 treasuryAmt);
    event GovernanceUpdated(address indexed newGovernance);
    event CertificateUpdated(address indexed newCertificate);
    event MintingUpdated(address indexed newMinting);
    event StakingUpdated(address indexed newStaking);
    event TreasuryUpdated(address indexed newTreasury);
    event MaxBenchmarkPoolWeiUpdated(uint256 newMax);

    modifier onlyGovernance() { require(msg.sender == governance, "PWMRewardERC20: not governance"); _; }
    modifier onlyCertificate() { require(msg.sender == certificate, "PWMRewardERC20: not certificate"); _; }

    constructor(address token_, address initialGovernance) {
        require(token_ != address(0), "PWMRewardERC20: zero token");
        require(initialGovernance != address(0), "PWMRewardERC20: zero governance");
        pwmToken   = IERC20(token_);
        governance = initialGovernance;
    }

    // ---------- governance ----------

    function setGovernance(address x) external onlyGovernance {
        require(x != address(0), "PWMRewardERC20: zero governance");
        governance = x;
        emit GovernanceUpdated(x);
    }
    function setCertificate(address x) external onlyGovernance {
        require(x != address(0), "PWMRewardERC20: zero certificate");
        certificate = x;
        emit CertificateUpdated(x);
    }
    function setMinting(address x) external onlyGovernance {
        require(x != address(0), "PWMRewardERC20: zero minting");
        minting = x;
        emit MintingUpdated(x);
    }
    function setStaking(address x) external onlyGovernance {
        require(x != address(0), "PWMRewardERC20: zero staking");
        staking = x;
        emit StakingUpdated(x);
    }
    function setTreasury(address x) external onlyGovernance {
        require(x != address(0), "PWMRewardERC20: zero treasury");
        treasury = IPWMTreasuryERC20(x);
        emit TreasuryUpdated(x);
    }
    function setMaxBenchmarkPoolWei(uint256 newMax) external onlyGovernance {
        require(newMax > 0, "PWMRewardERC20: zero max disables cap; use governance proposal explicitly");
        maxBenchmarkPoolWei = newMax;
        emit MaxBenchmarkPoolWeiUpdated(newMax);
    }

    // ---------- pool inflows ----------

    function seedBPool(bytes32 benchmarkHash, uint256 amount) external {
        require(msg.sender == staking, "PWMRewardERC20: not staking");
        pwmToken.safeTransferFrom(msg.sender, address(this), amount);
        _credit(benchmarkHash, amount, "B-pool-seed");
    }

    function depositMinting(bytes32 benchmarkHash, uint256 amount) external {
        require(msg.sender == minting, "PWMRewardERC20: not minting");
        pwmToken.safeTransferFrom(msg.sender, address(this), amount);
        _credit(benchmarkHash, amount, "A-pool-minting");
    }

    function depositBounty(bytes32 benchmarkHash, uint256 amount) external onlyGovernance {
        pwmToken.safeTransferFrom(msg.sender, address(this), amount);
        _credit(benchmarkHash, amount, "B-pool-bounty");
    }

    function _credit(bytes32 benchmarkHash, uint256 amount, string memory kind) internal {
        require(benchmarkHash != bytes32(0), "PWMRewardERC20: zero benchmark");
        require(amount > 0, "PWMRewardERC20: zero amount");
        uint256 newBalance = pool[benchmarkHash] + amount;
        uint256 cap = maxBenchmarkPoolWei;
        if (cap != 0) {
            require(newBalance <= cap, "PWMRewardERC20: pool cap exceeded");
        }
        pool[benchmarkHash] = newBalance;
        emit PoolSeeded(benchmarkHash, amount, newBalance, msg.sender, kind);
    }

    // ---------- settlement ----------

    function rankBps(uint8 rank) public pure returns (uint16) {
        if (rank == 1) return 4_000;
        if (rank == 2) return 500;
        if (rank == 3) return 200;
        if (rank >= 4 && rank <= MAX_RANK) return 100;
        return 0;
    }

    function distribute(bytes32 certHash, Draw calldata d) external onlyCertificate {
        require(!settled[certHash], "PWMRewardERC20: already settled");
        require(d.shareRatioP >= 1000 && d.shareRatioP <= 9000, "PWMRewardERC20: p out of range");
        require(d.acWallet != address(0) && d.cpWallet != address(0), "PWMRewardERC20: zero AC/CP");
        require(d.l1Creator != address(0) && d.l2Creator != address(0) && d.l3Creator != address(0),
                "PWMRewardERC20: zero creator");
        require(address(treasury) != address(0), "PWMRewardERC20: treasury unset");

        settled[certHash] = true;

        uint16 rbps = rankBps(d.rank);
        if (rbps == 0) {
            emit DrawSettled(certHash, d.benchmarkHash, d.rank, 0, pool[d.benchmarkHash]);
            return;
        }

        uint256 balance = pool[d.benchmarkHash];
        uint256 drawAmt = (balance * rbps) / BPS_DENOM;
        if (drawAmt == 0) {
            emit DrawSettled(certHash, d.benchmarkHash, d.rank, 0, balance);
            return;
        }

        pool[d.benchmarkHash] = balance - drawAmt;

        uint256 acAmt = (drawAmt * uint256(d.shareRatioP) * SPLIT_AC_CP) / (uint256(BPS_DENOM) * BPS_DENOM);
        uint256 cpAmt = (drawAmt * (BPS_DENOM - uint256(d.shareRatioP)) * SPLIT_AC_CP) / (uint256(BPS_DENOM) * BPS_DENOM);
        uint256 l3Amt = (drawAmt * SPLIT_L3) / BPS_DENOM;
        uint256 l2Amt = (drawAmt * SPLIT_L2) / BPS_DENOM;
        uint256 l1Amt = (drawAmt * SPLIT_L1) / BPS_DENOM;
        uint256 tkAmt = drawAmt - acAmt - cpAmt - l3Amt - l2Amt - l1Amt;

        if (acAmt > 0) pwmToken.safeTransfer(d.acWallet, acAmt);
        if (cpAmt > 0) pwmToken.safeTransfer(d.cpWallet, cpAmt);
        if (l3Amt > 0) pwmToken.safeTransfer(d.l3Creator, l3Amt);
        if (l2Amt > 0) pwmToken.safeTransfer(d.l2Creator, l2Amt);
        if (l1Amt > 0) pwmToken.safeTransfer(d.l1Creator, l1Amt);
        if (tkAmt > 0) {
            pwmToken.forceApprove(address(treasury), tkAmt);
            treasury.receive15pct(d.principleId, tkAmt);
        }

        emit RoyaltiesPaid(certHash, d.acWallet, acAmt, d.cpWallet, cpAmt, tkAmt);
        emit DrawSettled(certHash, d.benchmarkHash, d.rank, drawAmt, pool[d.benchmarkHash]);
    }

    function poolOf(bytes32 benchmarkHash) external view returns (uint256) {
        return pool[benchmarkHash];
    }
}
