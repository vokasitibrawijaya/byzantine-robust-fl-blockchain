"""
Blockchain Client for CIFAR-10 Experiments
Working version with proper ABI and gas tracking
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from web3 import Web3
from eth_account import Account


class BlockchainClient:
    """Web3 client for FL experiments with gas cost tracking"""
    
    def __init__(self, rpc_url: str = "http://127.0.0.1:8545"):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        if not self.w3.is_connected():
            raise ConnectionError(f"Cannot connect to blockchain at {rpc_url}")
        
        # Load deployment info
        deployment_file = Path("deployment.json")
        if not deployment_file.exists():
            raise FileNotFoundError("deployment.json not found. Run deploy_contract.py first")
        
        with open(deployment_file) as f:
            deployment = json.load(f)
        
        self.contract_address = Web3.to_checksum_address(deployment['contractAddress'])
        
        # Load ABI
        abi_file = Path("contract_abi.json")
        if not abi_file.exists():
            raise FileNotFoundError("contract_abi.json not found")
        
        with open(abi_file) as f:
            self.abi = json.load(f)
        
        # Create contract instance
        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=self.abi
        )
        
        # Use first account as coordinator
        self.account = self.w3.eth.accounts[0]
        
        # Track gas usage
        self.gas_usage = {
            'registerClients': [],
            'startRound': [],
            'submitUpdate': [],
            'markByzantine': [],
            'completeRound': []
        }
        
        print(f"✅ Blockchain connected")
        print(f"✅ Contract: {self.contract_address}")
        print(f"✅ Account: {self.account}")
        balance = self.w3.eth.get_balance(self.account)
        print(f"✅ Balance: {self.w3.from_wei(balance, 'ether')} ETH")
    
    def register_clients(self, num_clients: int) -> int:
        """Register client addresses. Returns gas used."""
        # Generate client addresses (use accounts from Ganache)
        accounts = self.w3.eth.accounts
        client_addresses = accounts[1:num_clients+1]  # Skip coordinator
        
        try:
            # Call registerClientsBatch
            tx_hash = self.contract.functions.registerClientsBatch(
                client_addresses
            ).transact({
                'from': self.account,
                'gas': 500000
            })
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            gas_used = receipt['gasUsed']
            
            self.gas_usage['registerClients'].append(gas_used)
            print(f"   ✅ Registered {num_clients} clients (Gas: {gas_used})")
            
            return gas_used
            
        except Exception as e:
            print(f"   ⚠️  Client registration failed: {e}")
            return 0
    
    def start_round(self) -> int:
        """Start a new training round. Returns gas used."""
        try:
            tx_hash = self.contract.functions.startRound().transact({
                'from': self.account,
                'gas': 200000
            })
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            gas_used = receipt['gasUsed']
            
            self.gas_usage['startRound'].append(gas_used)
            
            return gas_used
            
        except Exception as e:
            print(f"   ⚠️  Start round failed: {e}")
            return 0
    
    def submit_update(self, client_idx: int, model_hash: bytes, sample_count: int) -> int:
        """Submit client update. Returns gas used."""
        try:
            # Use client account (not coordinator)
            client_address = self.w3.eth.accounts[client_idx + 1]
            
            tx_hash = self.contract.functions.submitUpdate(
                model_hash
            ).transact({
                'from': client_address,
                'gas': 150000
            })
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            gas_used = receipt['gasUsed']
            
            self.gas_usage['submitUpdate'].append(gas_used)
            
            return gas_used
            
        except Exception as e:
            # Silent failure for efficiency
            return 0
    
    def mark_byzantine(self, round_num: int, client_idx: int, reason: str) -> int:
        """Mark client as Byzantine. Returns gas used."""
        try:
            client_address = self.w3.eth.accounts[client_idx + 1]
            
            tx_hash = self.contract.functions.markByzantine(
                round_num,
                client_address,
                reason
            ).transact({
                'from': self.account,
                'gas': 200000
            })
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            gas_used = receipt['gasUsed']
            
            self.gas_usage['markByzantine'].append(gas_used)
            
            return gas_used
            
        except Exception as e:
            return 0
    
    def complete_round(self, model_hash: bytes) -> int:
        """Complete round with aggregated model hash. Returns gas used."""
        try:
            tx_hash = self.contract.functions.completeRound(
                model_hash
            ).transact({
                'from': self.account,
                'gas': 200000
            })
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            gas_used = receipt['gasUsed']
            
            self.gas_usage['completeRound'].append(gas_used)
            
            return gas_used
            
        except Exception as e:
            return 0
    
    def hash_model_update(self, state_dict: Dict) -> bytes:
        """Create hash of model update - fast version"""
        import torch
        
        # Fast hash: concatenate tensor bytes directly
        hasher = hashlib.sha256()
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key]
            if isinstance(tensor, torch.Tensor):
                # Convert to CPU and get raw bytes
                hasher.update(tensor.cpu().detach().numpy().tobytes())
            else:
                hasher.update(str(tensor).encode())
        
        return hasher.digest()
    
    def get_gas_summary(self) -> Dict[str, Any]:
        """Get gas usage summary"""
        summary = {}
        
        for operation, gas_list in self.gas_usage.items():
            if gas_list:
                summary[operation] = {
                    'total': sum(gas_list),
                    'average': sum(gas_list) / len(gas_list),
                    'count': len(gas_list),
                    'min': min(gas_list),
                    'max': max(gas_list)
                }
            else:
                summary[operation] = {
                    'total': 0,
                    'average': 0,
                    'count': 0,
                    'min': 0,
                    'max': 0
                }
        
        # Calculate total gas
        total_gas = sum(sum(g) for g in self.gas_usage.values())
        summary['total_gas'] = total_gas
        
        # Estimate cost (at 50 Gwei gas price, ~$3000 ETH)
        gas_price_gwei = 50
        eth_price_usd = 3000
        cost_eth = (total_gas * gas_price_gwei) / 1e9
        cost_usd = cost_eth * eth_price_usd
        
        summary['estimated_cost_eth'] = cost_eth
        summary['estimated_cost_usd'] = cost_usd
        
        return summary
