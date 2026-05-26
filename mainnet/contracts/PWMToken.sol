// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {ERC20Capped} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Capped.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";

/// @title PWMToken
/// @notice Fixed-supply ERC20 token for the PWM protocol.
///
/// Supply: 21,000,000 PWM (hard cap, enforced by ERC20Capped).
/// At deployment, the entire 21M supply is minted to `initialHolder` in a
/// single mint call. After that, the owner can call mint() only up to the
/// cap (which is already reached at genesis), so the practical contract
/// behavior is "non-inflationary after deployment."
///
/// Genesis allocation (executed off-chain by the initial holder, typically a
/// 3-of-5 multisig, immediately after deploy):
///   17_220_000 PWM (82%) → PWMMintingERC20 (locked, emits per epoch)
///    2_100_000 PWM (10%) → Reserve multisig (4-of-7)
///    1_050_000 PWM ( 5%) → Uniswap v3 PWM/USDC pool (protocol-owned liquidity)
///      630_000 PWM ( 3%) → PWMVesting (founding team, 12mo cliff + 48mo linear)
///
/// EIP-2612 permit() is intentionally deferred — its OZ implementation
/// requires Solidity 0.8.25+ for the mcopy opcode. Approval-then-stake is
/// the supported flow until paymaster (Phase 3).
contract PWMToken is ERC20, ERC20Capped, Ownable {
    uint256 public constant TOTAL_SUPPLY = 21_000_000 ether;

    event GenesisMinted(address indexed to, uint256 amount);

    constructor(address initialHolder, address initialOwner)
        ERC20("Physics World Model", "PWM")
        ERC20Capped(TOTAL_SUPPLY)
        Ownable(initialOwner)
    {
        require(initialHolder != address(0), "PWMToken: zero holder");
        _mint(initialHolder, TOTAL_SUPPLY);
        emit GenesisMinted(initialHolder, TOTAL_SUPPLY);
    }

    /// @dev OZ v5 multi-inheritance: ERC20 + ERC20Capped both override _update.
    function _update(address from, address to, uint256 value)
        internal
        override(ERC20, ERC20Capped)
    {
        super._update(from, to, value);
    }
}
