// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract FLAudit {
    address public immutable coordinator;

    struct RoundSummary {
        bytes32 aggregateHash;
        uint256 submittedCount;
        uint256 flaggedCount;
        uint256 trimCountEachTail;
        uint256 timestamp;
    }

    mapping(uint256 => RoundSummary) public roundSummaries;
    mapping(uint256 => mapping(uint256 => bytes32)) public updateHashes;
    mapping(uint256 => mapping(uint256 => bool)) public flaggedUpdates;
    mapping(uint256 => mapping(uint256 => bool)) public updateRecorded;
    mapping(uint256 => bool) public roundFinalized;

    event ClientUpdateRecorded(
        uint256 indexed roundNumber,
        uint256 indexed clientId,
        bytes32 updateHash,
        bool flagged
    );
    event RoundFinalized(
        uint256 indexed roundNumber,
        bytes32 aggregateHash,
        uint256 submittedCount,
        uint256 flaggedCount,
        uint256 trimCountEachTail
    );

    constructor() {
        coordinator = msg.sender;
    }

    modifier onlyCoordinator() {
        require(msg.sender == coordinator, "coordinator only");
        _;
    }

    function recordClientUpdate(
        uint256 roundNumber,
        uint256 clientId,
        bytes32 updateHash,
        bool flagged
    ) external onlyCoordinator {
        require(
            !updateRecorded[roundNumber][clientId],
            "client update already recorded"
        );
        updateRecorded[roundNumber][clientId] = true;
        updateHashes[roundNumber][clientId] = updateHash;
        flaggedUpdates[roundNumber][clientId] = flagged;
        emit ClientUpdateRecorded(roundNumber, clientId, updateHash, flagged);
    }

    function finalizeRound(
        uint256 roundNumber,
        bytes32 aggregateHash,
        uint256 submittedCount,
        uint256 flaggedCount,
        uint256 trimCountEachTail
    ) external onlyCoordinator {
        require(!roundFinalized[roundNumber], "round already finalized");
        require(flaggedCount <= submittedCount, "invalid flagged count");
        require(
            2 * trimCountEachTail < submittedCount,
            "invalid trim count"
        );
        roundFinalized[roundNumber] = true;
        roundSummaries[roundNumber] = RoundSummary({
            aggregateHash: aggregateHash,
            submittedCount: submittedCount,
            flaggedCount: flaggedCount,
            trimCountEachTail: trimCountEachTail,
            timestamp: block.timestamp
        });
        emit RoundFinalized(
            roundNumber,
            aggregateHash,
            submittedCount,
            flaggedCount,
            trimCountEachTail
        );
    }
}
