"""
Blockchain Client untuk CIFAR-10 Experiments
Simplified version untuk Docker deployment
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from web3 import Web3
from eth_account import Account

class BlockchainClient:
    """Simplified Web3 client for CIFAR-10 experiments"""
    
    def __init__(self, rpc_url: str, contract_address: str, private_key: str, abi_path: str = None):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        if not self.w3.is_connected():
            raise ConnectionError(f"Cannot connect to blockchain at {rpc_url}")
        
        self.account = Account.from_key(private_key)
        self.contract_address = Web3.to_checksum_address(contract_address)
        
        # Load contract ABI
        if abi_path and Path(abi_path).exists():
            with open(abi_path, 'r') as f:
                contract_json = json.load(f)
                self.abi = contract_json['abi']
        else:
            # Fallback: minimal ABI
            self.abi = [
                {
                    "inputs": [{"internalType": "address[]", "name": "clients", "type": "address[]"}],
                    "name": "registerClientsBatch",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [],
                    "name": "startRound",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [
                        {"internalType": "bytes32", "name": "updateHash", "type": "bytes32"},
                        {"internalType": "uint256", "name": "sampleCount", "type": "uint256"}
                    ],
                    "name": "submitUpdate",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                }
            ]
        
        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=self.abi
        )
        
        print(f"✅ Blockchain connected: {rpc_url}")
        print(f"✅ Contract: {self.contract_address}")
        print(f"✅ Account: {self.account.address}")
        print(f"✅ Balance: {self.w3.from_wei(self.w3.eth.get_balance(self.account.address), 'ether')} ETH")
    
    def register_clients_batch(self, client_addresses: List[str]) -> Dict[str, Any]:
        """Register multiple clients"""
        client_addresses = [Web3.to_checksum_address(addr) for addr in client_addresses]
        
        tx = self.contract.functions.registerClientsBatch(client_addresses).build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': 500000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        signed_tx = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            'success': receipt['status'] == 1,
            'tx_hash': tx_hash.hex(),
            'gas_used': receipt['gasUsed']
        }
    
    def start_round(self) -> Dict[str, Any]:
        """Start new FL round"""
        tx = self.contract.functions.startRound().build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        signed_tx = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            'success': receipt['status'] == 1,
            'tx_hash': tx_hash.hex(),
            'gas_used': receipt['gasUsed']
        }
    
    def submit_update(self, update_hash: str, sample_count: int) -> Dict[str, Any]:
        """Submit model update hash"""
        # Ensure hash is bytes32 format
        if not update_hash.startswith('0x'):
            update_hash = '0x' + update_hash
        
        tx = self.contract.functions.submitUpdate(
            Web3.to_bytes(hexstr=update_hash),
            sample_count
        ).build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': 300000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        signed_tx = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            'success': receipt['status'] == 1,
            'tx_hash': tx_hash.hex(),
            'gas_used': receipt['gasUsed']
        }


def create_client_accounts(num_clients: int) -> List[Dict[str, str]]:
    """Generate client accounts for FL"""
    accounts = []
    for i in range(num_clients):
        account = Account.create()
        accounts.append({
            'address': account.address,
            'private_key': account.key.hex()
        })
    return accounts
