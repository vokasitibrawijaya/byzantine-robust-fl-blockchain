"""Ganache-backed audit log for the unified federated-learning experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch
from web3 import Web3


def hash_tensors(tensors: Iterable[torch.Tensor]) -> bytes:
    hasher = hashlib.sha256()
    for tensor in tensors:
        contiguous = tensor.detach().cpu().contiguous()
        hasher.update(str(tuple(contiguous.shape)).encode("ascii"))
        hasher.update(str(contiguous.dtype).encode("ascii"))
        hasher.update(contiguous.numpy().tobytes())
    return hasher.digest()


class FLAuditChain:
    def __init__(
        self,
        artifact_directory: Path,
        rpc_url: str = "http://127.0.0.1:8545",
    ) -> None:
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise ConnectionError(f"Cannot connect to Ganache at {rpc_url}")

        abi_files = sorted(artifact_directory.glob("*FLAudit.abi"))
        bin_files = sorted(artifact_directory.glob("*FLAudit.bin"))
        if len(abi_files) != 1 or len(bin_files) != 1:
            raise FileNotFoundError(
                "Compile FLAudit.sol first; expected one .abi and one .bin file"
            )

        self.abi = json.loads(abi_files[0].read_text(encoding="utf-8"))
        self.bytecode = bin_files[0].read_text(encoding="utf-8").strip()
        self.coordinator = self.w3.eth.accounts[0]
        self.transactions: list[dict] = []

        factory = self.w3.eth.contract(abi=self.abi, bytecode=self.bytecode)
        deployment_hash = factory.constructor().transact(
            {"from": self.coordinator}
        )
        deployment_receipt = self.w3.eth.wait_for_transaction_receipt(
            deployment_hash
        )
        if deployment_receipt.status != 1:
            raise RuntimeError("FLAudit contract deployment failed")

        self.contract_address = deployment_receipt.contractAddress
        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=self.abi,
        )
        self.deployment = self._receipt_record(
            "deploy",
            deployment_hash,
            deployment_receipt,
        )

    def _receipt_record(self, operation: str, tx_hash, receipt) -> dict:
        block = self.w3.eth.get_block(receipt.blockNumber)
        return {
            "operation": operation,
            "tx_hash": tx_hash.hex(),
            "status": int(receipt.status),
            "block_number": int(receipt.blockNumber),
            "block_hash": block.hash.hex(),
            "gas_used": int(receipt.gasUsed),
        }

    def record_client_update(
        self,
        round_number: int,
        client_id: int,
        update_hash: bytes,
        flagged: bool,
    ) -> dict:
        tx_hash = self.contract.functions.recordClientUpdate(
            round_number,
            client_id,
            update_hash,
            flagged,
        ).transact({"from": self.coordinator})
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        record = self._receipt_record("record_client_update", tx_hash, receipt)
        record.update(
            {
                "round": round_number,
                "client_id": client_id,
                "update_hash": "0x" + update_hash.hex(),
                "flagged": flagged,
            }
        )
        self.transactions.append(record)
        return record

    def finalize_round(
        self,
        round_number: int,
        aggregate_hash: bytes,
        submitted_count: int,
        flagged_count: int,
        trim_count_each_tail: int,
    ) -> dict:
        tx_hash = self.contract.functions.finalizeRound(
            round_number,
            aggregate_hash,
            submitted_count,
            flagged_count,
            trim_count_each_tail,
        ).transact({"from": self.coordinator})
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        record = self._receipt_record("finalize_round", tx_hash, receipt)
        record.update(
            {
                "round": round_number,
                "aggregate_hash": "0x" + aggregate_hash.hex(),
                "submitted_count": submitted_count,
                "flagged_count": flagged_count,
                "trim_count_each_tail": trim_count_each_tail,
            }
        )
        self.transactions.append(record)
        return record

    def verify_client_update(
        self,
        round_number: int,
        client_id: int,
        expected_hash: bytes,
        expected_flagged: bool,
    ) -> bool:
        stored_hash = self.contract.functions.updateHashes(
            round_number,
            client_id,
        ).call()
        stored_flag = self.contract.functions.flaggedUpdates(
            round_number,
            client_id,
        ).call()
        stored_recorded = self.contract.functions.updateRecorded(
            round_number,
            client_id,
        ).call()
        return (
            stored_recorded
            and stored_hash == expected_hash
            and stored_flag == expected_flagged
        )

    def verify_round_summary(
        self,
        round_number: int,
        expected_hash: bytes,
        expected_submitted_count: int,
        expected_flagged_count: int,
        expected_trim_count_each_tail: int,
    ) -> bool:
        stored = self.contract.functions.roundSummaries(round_number).call()
        finalized = self.contract.functions.roundFinalized(round_number).call()
        return (
            finalized
            and stored[0] == expected_hash
            and int(stored[1]) == expected_submitted_count
            and int(stored[2]) == expected_flagged_count
            and int(stored[3]) == expected_trim_count_each_tail
            and int(stored[4]) > 0
        )

    def evidence(self) -> dict:
        gas_total = self.deployment["gas_used"] + sum(
            transaction["gas_used"] for transaction in self.transactions
        )
        return {
            "rpc_url": self.w3.provider.endpoint_uri,
            "chain_id": int(self.w3.eth.chain_id),
            "latest_block": int(self.w3.eth.block_number),
            "coordinator": self.coordinator,
            "contract_address": self.contract_address,
            "deployment": self.deployment,
            "transactions": self.transactions,
            "transaction_count": len(self.transactions),
            "total_gas_used_including_deployment": gas_total,
        }
