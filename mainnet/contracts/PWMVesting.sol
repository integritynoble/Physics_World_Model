// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/// @title PWMVesting
/// @notice Founding-team token vesting with cliff + linear schedule.
///
/// Schedule:
///   - 0 .. cliffSeconds:               nothing released
///   - cliffSeconds .. durationSeconds: linear release of total deposit
///   - after durationSeconds:           full deposit released
///
/// Defaults (per pwm_overview1.md §Token Distribution):
///   cliffSeconds    = 365 days  (12 months)
///   durationSeconds = 4 × 365 days (48 months total from start)
///
/// One contract is deployed per beneficiary. The beneficiary calls release()
/// at any time to pull whatever has vested but not yet been released.
/// Tokens are held in this contract until released.
///
/// No admin functions for changing schedule or beneficiary. This is intentional:
/// vesting must be cryptographically unstoppable to be useful as a founder lock.
contract PWMVesting {
    using SafeERC20 for IERC20;

    IERC20  public immutable token;
    address public immutable beneficiary;
    uint64  public immutable start;
    uint64  public immutable cliff;       // absolute timestamp, == start + cliffSeconds
    uint64  public immutable duration;    // total duration in seconds from start

    uint256 public released;

    event Released(uint256 amount);

    constructor(
        address token_,
        address beneficiary_,
        uint64  start_,
        uint64  cliffSeconds,
        uint64  durationSeconds
    ) {
        require(token_ != address(0), "PWMVesting: zero token");
        require(beneficiary_ != address(0), "PWMVesting: zero beneficiary");
        require(durationSeconds > 0, "PWMVesting: zero duration");
        require(cliffSeconds <= durationSeconds, "PWMVesting: cliff > duration");

        token       = IERC20(token_);
        beneficiary = beneficiary_;
        start       = start_;
        cliff       = start_ + cliffSeconds;
        duration    = durationSeconds;
    }

    /// @notice Release all vested-but-not-yet-released tokens to the beneficiary.
    function release() external {
        uint256 amount = releasable();
        require(amount > 0, "PWMVesting: nothing to release");
        released += amount;
        token.safeTransfer(beneficiary, amount);
        emit Released(amount);
    }

    /// @notice Amount currently vested-and-not-yet-released.
    function releasable() public view returns (uint256) {
        return _vestedAt(uint64(block.timestamp)) - released;
    }

    /// @notice Amount vested up to a given timestamp (inclusive of already-released).
    function vestedAt(uint64 timestamp) external view returns (uint256) {
        return _vestedAt(timestamp);
    }

    /// @notice Total deposit currently in the vesting contract (released + locked).
    function totalAllocation() public view returns (uint256) {
        return token.balanceOf(address(this)) + released;
    }

    function _vestedAt(uint64 timestamp) internal view returns (uint256) {
        if (timestamp < cliff) return 0;

        uint256 total = totalAllocation();
        if (timestamp >= start + duration) return total;

        // Linear release between cliff and start+duration.
        // (timestamp - start) / duration gives the elapsed fraction.
        return (total * (timestamp - start)) / duration;
    }
}
