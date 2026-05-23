// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/// @title PWMTreasuryERC20
/// @notice ERC20 sibling of PWMTreasury. Per-principle T_k bookkeeping.
///         Each principle is self-funded: accumulates 15% of every L4 draw
///         under that principle and pays out M4 adversarial bounties
///         (cap: 50% of current T_k).
contract PWMTreasuryERC20 {
    using SafeERC20 for IERC20;

    IERC20  public immutable pwmToken;
    address public governance;
    address public reward;

    mapping(uint256 => uint256) public treasury;

    // Soft-launch posture: T_k payouts are paused at deploy. Governance
    // unpauses after audit clears. Inflows via receive15pct are not gated
    // by this flag (they are gated by onlyReward + paused upstream minting
    // makes inflows null during the 30-day soft-launch).
    bool public transfersPaused;

    event FundsReceived(uint256 indexed principleId, uint256 amount, uint256 newBalance);
    event BountyPaid(uint256 indexed principleId, address indexed winner, uint256 amount, uint256 newBalance);
    event GovernanceUpdated(address indexed newGovernance);
    event RewardUpdated(address indexed newReward);
    event TransfersPausedUpdated(bool paused);

    modifier onlyGovernance() {
        require(msg.sender == governance, "PWMTreasuryERC20: not governance");
        _;
    }

    modifier onlyReward() {
        require(msg.sender == reward, "PWMTreasuryERC20: not reward");
        _;
    }

    constructor(address token_, address initialGovernance) {
        require(token_ != address(0), "PWMTreasuryERC20: zero token");
        require(initialGovernance != address(0), "PWMTreasuryERC20: zero governance");
        pwmToken   = IERC20(token_);
        governance = initialGovernance;
    }

    function setGovernance(address newGovernance) external onlyGovernance {
        require(newGovernance != address(0), "PWMTreasuryERC20: zero governance");
        governance = newGovernance;
        emit GovernanceUpdated(newGovernance);
    }

    function setReward(address newReward) external onlyGovernance {
        require(newReward != address(0), "PWMTreasuryERC20: zero reward");
        reward = newReward;
        emit RewardUpdated(newReward);
    }

    function setTransfersPaused(bool paused) external onlyGovernance {
        transfersPaused = paused;
        emit TransfersPausedUpdated(paused);
    }

    /// @notice Credit T_k with the 15% cut from a completed L4 draw.
    ///         Caller (PWMRewardERC20) must have approved this contract for `amount`.
    function receive15pct(uint256 principleId, uint256 amount) external onlyReward {
        require(amount > 0, "PWMTreasuryERC20: zero amount");
        pwmToken.safeTransferFrom(msg.sender, address(this), amount);
        treasury[principleId] += amount;
        emit FundsReceived(principleId, amount, treasury[principleId]);
    }

    /// @notice Pay an M4 adversarial bounty to `winner`, capped at 50% of T_k.
    function payAdversarialBounty(uint256 principleId, address winner, uint256 amount)
        external
        onlyGovernance
    {
        require(!transfersPaused, "PWMTreasuryERC20: transfers paused");
        require(winner != address(0), "PWMTreasuryERC20: zero winner");
        uint256 balance = treasury[principleId];
        require(amount > 0, "PWMTreasuryERC20: zero amount");
        require(amount <= balance, "PWMTreasuryERC20: exceeds balance");
        require(amount * 2 <= balance, "PWMTreasuryERC20: exceeds 50% cap");

        treasury[principleId] = balance - amount;
        pwmToken.safeTransfer(winner, amount);

        emit BountyPaid(principleId, winner, amount, treasury[principleId]);
    }

    function balanceOf(uint256 principleId) external view returns (uint256) {
        return treasury[principleId];
    }
}
