// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract FLAudit {
    address public immutable coordinator;

    struct RoundSummary {
        bytes32 aggregateHash;
        uint256 acceptedCount;
        uint256 flaggedCount;
        uint256 timestamp;
    }

    mapping(uint256 => RoundSummary) public roundSummaries;
    mapping(uint256 => mapping(uint256 => bytes32)) public updateHashes;
    mapping(uint256 => mapping(uint256 => bool)) public flaggedUpdates;

    event ClientUpdateRecorded(
        uint256 indexed roundNumber,
        uint256 indexed clientId,
        bytes32 updateHash,
        bool flagged
    );
    event RoundFinalized(
        uint256 indexed roundNumber,
        bytes32 aggregateHash,
        uint256 acceptedCount,
        uint256 flaggedCount
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
        updateHashes[roundNumber][clientId] = updateHash;
        flaggedUpdates[roundNumber][clientId] = flagged;
        emit ClientUpdateRecorded(roundNumber, clientId, updateHash, flagged);
    }

    function finalizeRound(
        uint256 roundNumber,
        bytes32 aggregateHash,
        uint256 acceptedCount,
        uint256 flaggedCount
    ) external onlyCoordinator {
        roundSummaries[roundNumber] = RoundSummary({
            aggregateHash: aggregateHash,
            acceptedCount: acceptedCount,
            flaggedCount: flaggedCount,
            timestamp: block.timestamp
        });
        emit RoundFinalized(
            roundNumber,
            aggregateHash,
            acceptedCount,
            flaggedCount
        );
    }
}
