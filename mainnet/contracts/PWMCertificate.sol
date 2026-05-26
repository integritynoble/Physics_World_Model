// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IPWMRegistry {
    function getArtifact(bytes32 hash) external view returns (bytes32, uint8, address, uint256);
    function exists(bytes32 hash) external view returns (bool);
}

interface IPWMReward {
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
    function distribute(bytes32 certHash, Draw calldata d) external;
}

interface IPWMMinting {
    function mintFor(uint256 principleId, bytes32 benchmarkHash) external returns (uint256);
}

/// @title PWMCertificate
/// @notice L4 certificate submission + challenge window + ranked-draw settlement.
///         Submission freezes payout wiring (AC/CP wallets, p); finalize() after the
///         challenge window dispatches to PWMReward. Challenges within the window
///         block finalization until governance resolves them.
contract PWMCertificate {
    uint256 public constant CHALLENGE_PERIOD_STANDARD = 7 days;
    uint256 public constant CHALLENGE_PERIOD_EXTENDED = 14 days;
    uint8   public constant DELTA_EXTEND_THRESHOLD    = 10;

    address public governance;
    IPWMRegistry public registry;
    IPWMReward   public reward;
    IPWMMinting  public minting;    // optional: if unset, finalize skips mintFor

    // Soft-launch posture: submissions are gated to a governance-approved
    // allow-list until the protocol matures and audit clears. Setting
    // submissionPermissionless = true opens submit() to all callers.
    mapping(address => bool) public approvedSubmitter;
    bool public submissionPermissionless;

    enum Status { None, Pending, Challenged, Finalized, Rejected }

    struct Certificate {
        bytes32 benchmarkHash;
        uint256 principleId;
        address l1Creator;
        address l2Creator;
        address l3Creator;
        address acWallet;
        address cpWallet;
        uint16  shareRatioP;     // p × 10000, 1000..9000
        uint8   Q_int;
        uint8   delta;           // delta tier (0..255). ≥10 → extended window.
        uint8   rank;            // 1..10 for draw; 0/11+ treated as rollover by reward.
        address submitter;
        uint256 submittedAt;
        Status  status;
        address challenger;      // zero if not challenged
    }
    mapping(bytes32 => Certificate) public certificates;

    event CertificateSubmitted(
        bytes32 indexed certHash,
        bytes32 indexed benchmarkHash,
        address indexed submitter,
        uint8           Q_int
    );
    event CertificateChallenged(bytes32 indexed certHash, address indexed challenger, bytes proof);
    event ChallengeResolved(bytes32 indexed certHash, bool upheld);
    event CertificateFinalized(bytes32 indexed certHash, uint8 rank);
    event GovernanceUpdated(address indexed newGovernance);
    event RegistryUpdated(address indexed newRegistry);
    event RewardUpdated(address indexed newReward);
    event MintingUpdated(address indexed newMinting);
    event ApprovedSubmitterUpdated(address indexed submitter, bool approved);
    event SubmissionPermissionlessUpdated(bool isPermissionless);

    modifier onlyGovernance() { require(msg.sender == governance, "PWMCertificate: not governance"); _; }
    modifier onlySubmitter() {
        require(submissionPermissionless || approvedSubmitter[msg.sender],
                "PWMCertificate: not approved submitter");
        _;
    }

    constructor(address initialGovernance) {
        require(initialGovernance != address(0), "PWMCertificate: zero governance");
        governance = initialGovernance;
    }

    // ---------- governance ----------

    function setGovernance(address x) external onlyGovernance {
        require(x != address(0), "PWMCertificate: zero governance");
        governance = x;
        emit GovernanceUpdated(x);
    }
    function setRegistry(address x) external onlyGovernance {
        require(x != address(0), "PWMCertificate: zero registry");
        registry = IPWMRegistry(x);
        emit RegistryUpdated(x);
    }
    function setReward(address x) external onlyGovernance {
        require(x != address(0), "PWMCertificate: zero reward");
        reward = IPWMReward(x);
        emit RewardUpdated(x);
    }
    function setMinting(address x) external onlyGovernance {
        require(x != address(0), "PWMCertificate: zero minting");
        minting = IPWMMinting(x);
        emit MintingUpdated(x);
    }

    function setApprovedSubmitter(address submitter, bool approved) external onlyGovernance {
        require(submitter != address(0), "PWMCertificate: zero submitter");
        approvedSubmitter[submitter] = approved;
        emit ApprovedSubmitterUpdated(submitter, approved);
    }

    function setSubmissionPermissionless(bool isPermissionless) external onlyGovernance {
        submissionPermissionless = isPermissionless;
        emit SubmissionPermissionlessUpdated(isPermissionless);
    }

    // ---------- submission ----------

    struct SubmitArgs {
        bytes32 certHash;
        bytes32 benchmarkHash;
        uint256 principleId;
        address l1Creator;
        address l2Creator;
        address l3Creator;
        address acWallet;
        address cpWallet;
        uint16  shareRatioP;
        uint8   Q_int;
        uint8   delta;
        uint8   rank;
    }

    function submit(SubmitArgs calldata a) external onlySubmitter {
        require(a.certHash != bytes32(0), "PWMCertificate: zero cert");
        require(certificates[a.certHash].status == Status.None, "PWMCertificate: already submitted");
        require(a.shareRatioP >= 1000 && a.shareRatioP <= 9000, "PWMCertificate: p out of range");
        require(a.acWallet != address(0) && a.cpWallet != address(0), "PWMCertificate: zero AC/CP");
        require(a.l1Creator != address(0) && a.l2Creator != address(0) && a.l3Creator != address(0),
                "PWMCertificate: zero creator");
        require(address(registry) != address(0), "PWMCertificate: registry unset");
        require(registry.exists(a.benchmarkHash), "PWMCertificate: benchmark not registered");

        certificates[a.certHash] = Certificate({
            benchmarkHash: a.benchmarkHash,
            principleId:   a.principleId,
            l1Creator:     a.l1Creator,
            l2Creator:     a.l2Creator,
            l3Creator:     a.l3Creator,
            acWallet:      a.acWallet,
            cpWallet:      a.cpWallet,
            shareRatioP:   a.shareRatioP,
            Q_int:         a.Q_int,
            delta:         a.delta,
            rank:          a.rank,
            submitter:     msg.sender,
            submittedAt:   block.timestamp,
            status:        Status.Pending,
            challenger:    address(0)
        });

        emit CertificateSubmitted(a.certHash, a.benchmarkHash, msg.sender, a.Q_int);
    }

    // ---------- challenge window ----------

    /// @notice File a challenge within the active window. Blocks finalize() until
    ///         governance resolves via resolveChallenge().
    function challenge(bytes32 certHash, bytes calldata proof) external {
        Certificate storage c = certificates[certHash];
        require(c.status == Status.Pending, "PWMCertificate: not pending");
        require(block.timestamp <= c.submittedAt + _windowOf(c.delta), "PWMCertificate: window closed");
        c.status = Status.Challenged;
        c.challenger = msg.sender;
        emit CertificateChallenged(certHash, msg.sender, proof);
    }

    /// @notice Governance resolves a challenge. `upheld=true` rejects the cert;
    ///         `upheld=false` reinstates it for finalization.
    function resolveChallenge(bytes32 certHash, bool upheld) external onlyGovernance {
        Certificate storage c = certificates[certHash];
        require(c.status == Status.Challenged, "PWMCertificate: not challenged");
        if (upheld) {
            c.status = Status.Rejected;
        } else {
            c.status = Status.Pending;
        }
        emit ChallengeResolved(certHash, upheld);
    }

    // ---------- finalize ----------

    /// @notice After the challenge window with no open challenge, dispatch to PWMReward.
    function finalize(bytes32 certHash) external {
        Certificate storage c = certificates[certHash];
        require(c.status == Status.Pending, "PWMCertificate: not pending");
        require(block.timestamp > c.submittedAt + _windowOf(c.delta), "PWMCertificate: window open");
        require(address(reward) != address(0), "PWMCertificate: reward unset");

        c.status = Status.Finalized;

        // Inject this event's share of protocol minting (Zeno A_{k,j,b}) into
        // the benchmark pool before the rank draw runs. Skipped if minting is
        // unset (test harness / pre-promotion) so the reward still distributes.
        //
        // SECURITY_REVIEW CC-2 fix: wrap in try/catch so finalize remains
        // operational even when minting is paused (soft-launch), the principle
        // isn't promoted, or any other transient mintFor revert. The reward
        // distribution proceeds against the (possibly un-incremented) pool
        // balance — submitters still get their rank-based draw from existing
        // pool seed even if this finalize-step contribution reverts.
        if (address(minting) != address(0)) {
            try minting.mintFor(c.principleId, c.benchmarkHash) returns (uint256) {} catch {}
        }

        IPWMReward.Draw memory d = IPWMReward.Draw({
            benchmarkHash: c.benchmarkHash,
            principleId:   c.principleId,
            l1Creator:     c.l1Creator,
            l2Creator:     c.l2Creator,
            l3Creator:     c.l3Creator,
            acWallet:      c.acWallet,
            cpWallet:      c.cpWallet,
            shareRatioP:   c.shareRatioP,
            rank:          c.rank
        });
        reward.distribute(certHash, d);

        emit CertificateFinalized(certHash, c.rank);
    }

    // ---------- views ----------

    function windowEndOf(bytes32 certHash) external view returns (uint256) {
        Certificate storage c = certificates[certHash];
        if (c.submittedAt == 0) return 0;
        return c.submittedAt + _windowOf(c.delta);
    }

    function _windowOf(uint8 delta) internal pure returns (uint256) {
        return delta >= DELTA_EXTEND_THRESHOLD ? CHALLENGE_PERIOD_EXTENDED : CHALLENGE_PERIOD_STANDARD;
    }
}
