// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title PWMGovernance
/// @notice Phase 1-2: 3-of-5 multisig over the 5 founder wallets. Proposes and
///         confirms parameter changes behind a 48-hour time-lock. activateDAO()
///         is a one-way switch that disables the multisig path forever; the
///         DAO voting implementation itself is deferred (post-M3 per roadmap).
contract PWMGovernance {
    uint256 public constant REQUIRED_APPROVALS = 3;
    uint256 public constant NUM_FOUNDERS       = 5;
    uint256 public constant TIME_LOCK          = 48 hours;

    address[NUM_FOUNDERS] public founders;
    mapping(address => bool) public isFounder;

    mapping(bytes32 => uint256) public parameters; // key => current value

    struct Proposal {
        bytes32 key;
        uint256 value;
        uint256 proposedAt;    // 0 once executed/cancelled
        // thresholdReachedAt: block.timestamp of the approveProposal call that
        // first brought approvals to REQUIRED_APPROVALS (3-of-5). Used as the
        // start of the 48h timelock for executeProposal and activateDAO instead
        // of proposedAt. Symmetric with ExecProposal.thresholdReachedAt — see
        // SECURITY_REVIEW M-1 / A1-v3 carry-over: with timelock-from-proposedAt,
        // three colluders could approve in the last second of the 48h window,
        // leaving non-colluding founders no realistic veto window. Now the 48h
        // is measured from "quorum reached" so non-colluding founders always
        // have a full 48h to cancelProposal after the proposal becomes
        // executable. Defaults to 0; only set once.
        uint256 thresholdReachedAt;
        uint8   approvals;     // count of yeses
        bool    executed;
        bool    cancelled;
    }
    mapping(uint256 => Proposal) public proposals;
    mapping(uint256 => mapping(address => bool)) public approved; // proposalId => founder => yes?
    uint256 public nextProposalId;

    // ---- Founder-change proposals (added 2026-05-11; PWM_FOUNDER_ROTATION_2026-05-10.md Path A) ----

    struct FounderChange {
        uint8   slot;        // 0..NUM_FOUNDERS-1
        address newAddr;
        uint256 proposedAt;  // (legacy; kept for backwards-compatibility with storage layout)
        // thresholdReachedAt: same semantics as Proposal.thresholdReachedAt and
        // ExecProposal.thresholdReachedAt — block.timestamp of the approval that
        // first brought approvals to REQUIRED_APPROVALS. Used as the start of
        // the 48h timelock for executeFounderChange instead of proposedAt.
        // SECURITY_REVIEW M-2 / A1-v3 carry-over fix.
        uint256 thresholdReachedAt;
        uint8   approvals;
        bool    executed;
        bool    cancelled;
    }
    mapping(uint256 => FounderChange) public founderChangeProposals;
    mapping(uint256 => mapping(address => bool)) public approvedFounderChange;
    uint256 public nextFounderChangeId;

    // ---- Arbitrary-call exec proposals (added 2026-05-18; A1-v2 CRITICAL fix) ----
    //
    // Without this, PWMGovernance can only write to its internal `parameters`
    // mapping. Sibling contracts (PWMStakingERC20, PWMRewardERC20, PWMTreasuryERC20,
    // PWMMintingERC20, PWMCertificate) gate their setters with `onlyGovernance`,
    // so once `setGovernance(address(this))` is called on them, every setX becomes
    // permanently unreachable. This primitive lets the same 3-of-5 + 48h timelock
    // gate arbitrary-target calls on those sibling contracts.

    struct ExecProposal {
        address target;
        bytes   data;
        uint256 proposedAt;
        // thresholdReachedAt: block.timestamp of the approveExec call that first
        // brought approvals to REQUIRED_APPROVALS (3-of-5). Used as the start of
        // the 48h timelock for executeExec instead of proposedAt. Fixes A1-v3
        // HIGH: with timelock-from-proposedAt, three colluders could approve in
        // the last second of the 48h window, leaving non-colluding founders no
        // realistic veto window. Now the 48h is measured from "quorum reached",
        // so non-colluding founders always have a full 48h to cancelExec after
        // the proposal becomes executable. Defaults to 0; only set once.
        uint256 thresholdReachedAt;
        uint8   approvals;
        bool    executed;
        bool    cancelled;
    }
    mapping(uint256 => ExecProposal) public execProposals;
    mapping(uint256 => mapping(address => bool)) public approvedExec;
    uint256 public nextExecProposalId;

    bool public daoActivated;

    event ProposalCreated(uint256 indexed id, bytes32 indexed key, uint256 value, address indexed proposer);
    event ProposalApproved(uint256 indexed id, address indexed founder, uint8 approvals);
    event ProposalExecuted(uint256 indexed id, bytes32 indexed key, uint256 value);
    event ProposalCancelled(uint256 indexed id);
    event DAOActivated(uint256 at);

    event FounderChangeProposed(uint256 indexed id, uint8 indexed slot, address indexed newAddr, address proposer);
    event FounderChangeApproved(uint256 indexed id, address indexed approver, uint8 approvals);
    event FounderChangeExecuted(uint256 indexed id, uint8 indexed slot, address oldAddr, address indexed newAddr);
    event FounderChangeCancelled(uint256 indexed id);

    event ExecProposalCreated(uint256 indexed id, address indexed target, bytes data, address indexed proposer);
    event ExecProposalApproved(uint256 indexed id, address indexed approver, uint8 approvals);
    event ExecProposalExecuted(uint256 indexed id, address indexed target, bytes returnData);
    event ExecProposalCancelled(uint256 indexed id);

    modifier onlyFounder() {
        require(isFounder[msg.sender], "PWMGovernance: not founder");
        _;
    }

    modifier multisigActive() {
        require(!daoActivated, "PWMGovernance: DAO active, multisig disabled");
        _;
    }

    constructor(address[NUM_FOUNDERS] memory _founders) {
        for (uint256 i = 0; i < NUM_FOUNDERS; i++) {
            require(_founders[i] != address(0), "PWMGovernance: zero founder");
            require(!isFounder[_founders[i]], "PWMGovernance: duplicate founder");
            founders[i] = _founders[i];
            isFounder[_founders[i]] = true;
        }
    }

    /// @notice Propose a parameter change. 48h timelock starts now.
    function proposeParameter(bytes32 key, uint256 value)
        external
        onlyFounder
        multisigActive
        returns (uint256 id)
    {
        id = nextProposalId++;
        proposals[id] = Proposal({
            key:                key,
            value:              value,
            proposedAt:         block.timestamp,
            thresholdReachedAt: 0,
            approvals:          1,
            executed:           false,
            cancelled:          false
        });
        approved[id][msg.sender] = true;
        emit ProposalCreated(id, key, value, msg.sender);
        emit ProposalApproved(id, msg.sender, 1);
    }

    /// @notice Approve a pending proposal (one vote per founder).
    /// @dev Records `thresholdReachedAt` on the approval that first brings
    ///      approvals to REQUIRED_APPROVALS — this is the start of the 48h
    ///      executeProposal/activateDAO timelock (NOT proposedAt). See
    ///      Proposal struct comment for the M-1 / A1-v3 carry-over rationale.
    function approveProposal(uint256 id) external onlyFounder multisigActive {
        Proposal storage p = proposals[id];
        require(p.proposedAt != 0, "PWMGovernance: unknown proposal");
        require(!p.executed && !p.cancelled, "PWMGovernance: finalised");
        require(!approved[id][msg.sender], "PWMGovernance: already approved");
        approved[id][msg.sender] = true;
        p.approvals += 1;
        // Set thresholdReachedAt the first time we cross the quorum threshold.
        // Subsequent approvals (4th, 5th) leave it unchanged so the timelock
        // window does not shift on late approvals.
        if (p.approvals == REQUIRED_APPROVALS) {
            p.thresholdReachedAt = block.timestamp;
        }
        emit ProposalApproved(id, msg.sender, p.approvals);
    }

    /// @notice Execute an approved proposal after the timelock elapses.
    function executeProposal(uint256 id) external onlyFounder multisigActive {
        Proposal storage p = proposals[id];
        require(p.proposedAt != 0, "PWMGovernance: unknown proposal");
        require(!p.executed && !p.cancelled, "PWMGovernance: finalised");
        require(p.approvals >= REQUIRED_APPROVALS, "PWMGovernance: insufficient approvals");
        // SECURITY_REVIEW M-1 fix: timelock from quorum-reached time, not from
        // propose-time. The `approvals >= REQUIRED_APPROVALS` guard above
        // implies thresholdReachedAt was set, so it is non-zero here.
        require(block.timestamp >= p.thresholdReachedAt + TIME_LOCK, "PWMGovernance: timelock not elapsed");

        p.executed = true;
        parameters[p.key] = p.value;
        emit ProposalExecuted(id, p.key, p.value);
    }

    /// @notice Any founder may cancel a pending proposal (e.g., if values are wrong).
    function cancelProposal(uint256 id) external onlyFounder multisigActive {
        Proposal storage p = proposals[id];
        require(p.proposedAt != 0, "PWMGovernance: unknown proposal");
        require(!p.executed && !p.cancelled, "PWMGovernance: finalised");
        p.cancelled = true;
        emit ProposalCancelled(id);
    }

    /// @notice Irreversible: transition to DAO voting. Requires 3-of-5 confirmation
    ///         via the same approval flow (here implemented as a dedicated entry point).
    ///         The DAO implementation itself is out-of-scope for M1.1; callers check
    ///         `daoActivated` and switch over the parameter-setting authority off-chain.
    function activateDAO(uint256 id) external onlyFounder multisigActive {
        Proposal storage p = proposals[id];
        require(p.proposedAt != 0, "PWMGovernance: unknown proposal");
        require(!p.executed && !p.cancelled, "PWMGovernance: finalised");
        require(p.key == keccak256("ACTIVATE_DAO"), "PWMGovernance: wrong proposal key");
        require(p.approvals >= REQUIRED_APPROVALS, "PWMGovernance: insufficient approvals");
        // SECURITY_REVIEW M-1 fix: same timelock-from-threshold as executeProposal.
        require(block.timestamp >= p.thresholdReachedAt + TIME_LOCK, "PWMGovernance: timelock not elapsed");

        p.executed = true;
        daoActivated = true;
        emit DAOActivated(block.timestamp);
    }

    /// @notice Read a parameter value (0 if unset).
    function getParameter(bytes32 key) external view returns (uint256) {
        return parameters[key];
    }

    // ---- Founder rotation (added 2026-05-11; PWM_FOUNDER_ROTATION_2026-05-10.md Path A) ----
    //
    // Mirrors the parameter-change pattern above: propose → 3-of-5 approvals →
    // 48h timelock → execute. Any founder can cancel a pending proposal.
    // Separate `nextFounderChangeId` counter and `approvedFounderChange` mapping
    // keep this id-space fully isolated from parameter proposals.
    //
    // Rationale: without rotation, founder-key loss or compromise has no
    // on-chain remediation after admin handoff to PWMGovernance — see
    // wallet/PWM_FOUNDER_ROTATION_2026-05-10.md "Critical correction: Path C
    // constraint" for the full analysis.

    /// @notice Propose swapping one founder slot to a new address. 48h timelock starts now.
    /// @param slot Index of the founder slot to replace (0..NUM_FOUNDERS-1).
    /// @param newAddr New address for that slot. Must be non-zero and not already a founder.
    /// @return id The id of the newly created founder-change proposal.
    function proposeFounderChange(uint8 slot, address newAddr)
        external
        onlyFounder
        multisigActive
        returns (uint256 id)
    {
        require(slot < NUM_FOUNDERS,    "PWMGovernance: invalid slot");
        require(newAddr != address(0),  "PWMGovernance: zero addr");
        require(!isFounder[newAddr],    "PWMGovernance: duplicate founder");

        id = nextFounderChangeId++;
        founderChangeProposals[id] = FounderChange({
            slot:               slot,
            newAddr:            newAddr,
            proposedAt:         block.timestamp,
            thresholdReachedAt: 0,
            approvals:          1,
            executed:           false,
            cancelled:          false
        });
        approvedFounderChange[id][msg.sender] = true;
        emit FounderChangeProposed(id, slot, newAddr, msg.sender);
        emit FounderChangeApproved(id, msg.sender, 1);
    }

    /// @notice Approve a pending founder-change proposal (one vote per founder).
    /// @dev Records `thresholdReachedAt` on the approval that first brings
    ///      approvals to REQUIRED_APPROVALS. See FounderChange struct comment.
    function approveFounderChange(uint256 id) external onlyFounder multisigActive {
        FounderChange storage p = founderChangeProposals[id];
        require(p.proposedAt != 0,                       "PWMGovernance: unknown proposal");
        require(!p.executed && !p.cancelled,             "PWMGovernance: finalised");
        require(!approvedFounderChange[id][msg.sender],  "PWMGovernance: already approved");
        approvedFounderChange[id][msg.sender] = true;
        p.approvals += 1;
        // Set thresholdReachedAt the first time we cross the quorum threshold.
        if (p.approvals == REQUIRED_APPROVALS) {
            p.thresholdReachedAt = block.timestamp;
        }
        emit FounderChangeApproved(id, msg.sender, p.approvals);
    }

    /// @notice Execute an approved founder swap after the 48h timelock elapses.
    /// @dev Re-checks newAddr is not already a founder (race condition: a prior
    ///      proposal may have set this same address while ours was pending).
    function executeFounderChange(uint256 id) external onlyFounder multisigActive {
        FounderChange storage p = founderChangeProposals[id];
        require(p.proposedAt != 0,                            "PWMGovernance: unknown proposal");
        require(!p.executed && !p.cancelled,                  "PWMGovernance: finalised");
        require(p.approvals >= REQUIRED_APPROVALS,            "PWMGovernance: insufficient approvals");
        // SECURITY_REVIEW M-2 fix: timelock from quorum-reached time, not from
        // propose-time.
        require(block.timestamp >= p.thresholdReachedAt + TIME_LOCK, "PWMGovernance: timelock not elapsed");
        require(!isFounder[p.newAddr],                        "PWMGovernance: duplicate founder");

        address oldAddr = founders[p.slot];
        founders[p.slot]     = p.newAddr;
        isFounder[oldAddr]   = false;
        isFounder[p.newAddr] = true;
        p.executed           = true;
        emit FounderChangeExecuted(id, p.slot, oldAddr, p.newAddr);
    }

    /// @notice Any founder may cancel a pending founder-change proposal
    ///         (e.g., if the proposer is suspected compromised or values are wrong).
    function cancelFounderChange(uint256 id) external onlyFounder multisigActive {
        FounderChange storage p = founderChangeProposals[id];
        require(p.proposedAt != 0,            "PWMGovernance: unknown proposal");
        require(!p.executed && !p.cancelled,  "PWMGovernance: finalised");
        p.cancelled = true;
        emit FounderChangeCancelled(id);
    }

    // ---- Arbitrary-call exec flow (A1-v2 CRITICAL fix) ----
    //
    // 3-of-5 + 48h timelock → arbitrary external call. Used to invoke setters
    // on sibling contracts that gate with `onlyGovernance`. Disallows targeting
    // this contract itself (a self-call would let governance re-enter and is
    // unnecessary because all internal mutations have dedicated entry points).
    // Disallows empty calldata (no plausible legitimate use case; reduces
    // accidental footguns from incorrectly-encoded transactions).

    /// @notice Propose an arbitrary external call. 48h timelock starts now.
    /// @param target Address to call. Must be non-zero and not this contract.
    /// @param data ABI-encoded calldata for the call. Must contain at least a
    ///        4-byte function selector.
    function proposeExec(address target, bytes calldata data)
        external
        onlyFounder
        multisigActive
        returns (uint256 id)
    {
        require(target != address(0),    "PWMGovernance: zero target");
        require(target != address(this), "PWMGovernance: cannot target self");
        require(data.length >= 4,        "PWMGovernance: data too short");

        id = nextExecProposalId++;
        execProposals[id] = ExecProposal({
            target:             target,
            data:               data,
            proposedAt:         block.timestamp,
            thresholdReachedAt: 0,
            approvals:          1,
            executed:           false,
            cancelled:          false
        });
        approvedExec[id][msg.sender] = true;
        emit ExecProposalCreated(id, target, data, msg.sender);
        emit ExecProposalApproved(id, msg.sender, 1);
    }

    /// @notice Approve a pending exec proposal (one vote per founder).
    /// @dev Records `thresholdReachedAt` on the approval that first brings
    ///      approvals to REQUIRED_APPROVALS — this is the start of the 48h
    ///      executeExec timelock (NOT proposedAt). See ExecProposal struct
    ///      comment for the A1-v3 HIGH rationale.
    function approveExec(uint256 id) external onlyFounder multisigActive {
        ExecProposal storage p = execProposals[id];
        require(p.proposedAt != 0,                "PWMGovernance: unknown exec proposal");
        require(!p.executed && !p.cancelled,      "PWMGovernance: finalised");
        require(!approvedExec[id][msg.sender],    "PWMGovernance: already approved");
        approvedExec[id][msg.sender] = true;
        p.approvals += 1;
        // Set thresholdReachedAt the first time we cross the quorum threshold.
        // Subsequent approvals (4th, 5th) leave it unchanged so the timelock
        // window does not shift on late approvals.
        if (p.approvals == REQUIRED_APPROVALS) {
            p.thresholdReachedAt = block.timestamp;
        }
        emit ExecProposalApproved(id, msg.sender, p.approvals);
    }

    /// @notice Execute an approved exec proposal after the timelock elapses.
    /// @dev CEI: sets p.executed = true BEFORE the external call so reentrant
    ///      callers cannot re-execute the same proposal. Bubbles up the revert
    ///      reason from the target on failure (the entire tx reverts, so
    ///      p.executed = true is rolled back — founders can re-attempt).
    function executeExec(uint256 id) external onlyFounder multisigActive {
        ExecProposal storage p = execProposals[id];
        require(p.proposedAt != 0,                            "PWMGovernance: unknown exec proposal");
        require(!p.executed && !p.cancelled,                  "PWMGovernance: finalised");
        require(p.approvals >= REQUIRED_APPROVALS,            "PWMGovernance: insufficient approvals");
        // A1-v3 HIGH fix: timelock measured from quorum-reached time, not from
        // propose-time. Non-colluding founders are guaranteed a full 48h veto
        // window after the proposal becomes executable. Note: approvals
        // >= REQUIRED_APPROVALS implies thresholdReachedAt was set by the
        // approveExec that crossed the threshold, so it is non-zero here.
        require(block.timestamp >= p.thresholdReachedAt + TIME_LOCK, "PWMGovernance: timelock not elapsed");

        p.executed = true;
        (bool success, bytes memory returnData) = p.target.call(p.data);
        if (!success) {
            if (returnData.length > 0) {
                // Bubble up the revert reason from the target.
                assembly {
                    let sz := mload(returnData)
                    revert(add(32, returnData), sz)
                }
            } else {
                revert("PWMGovernance: exec call failed (no reason)");
            }
        }
        emit ExecProposalExecuted(id, p.target, returnData);
    }

    /// @notice Any founder may cancel a pending exec proposal.
    function cancelExec(uint256 id) external onlyFounder multisigActive {
        ExecProposal storage p = execProposals[id];
        require(p.proposedAt != 0,            "PWMGovernance: unknown exec proposal");
        require(!p.executed && !p.cancelled,  "PWMGovernance: finalised");
        p.cancelled = true;
        emit ExecProposalCancelled(id);
    }
}
