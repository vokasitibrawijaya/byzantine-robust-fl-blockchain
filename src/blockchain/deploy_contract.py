"""
Deploy FederatedLearningAggregator smart contract to Ganache
"""
import json
from web3 import Web3
from pathlib import Path
from solcx import compile_source, install_solc
import sys

# Ensure solc compiler
print("📦 Installing Solidity compiler...")
try:
    install_solc('0.8.19')
    from solcx import set_solc_version
    set_solc_version('0.8.19')
    print("   ✅ Solc 0.8.19 installed")
except Exception as e:
    print(f"   ⚠️  Warning: {e}")

# Connect to Ganache
RPC_URL = "http://127.0.0.1:8545"
w3 = Web3(Web3.HTTPProvider(RPC_URL))

if not w3.is_connected():
    print("❌ Cannot connect to Ganache. Is it running?")
    print("Run: docker run -d -p 8545:8545 --name ganache-cifar10 trufflesuite/ganache:latest")
    exit(1)

print(f"\n✅ Connected to Ganache")
print(f"   Chain ID: {w3.eth.chain_id}")
print(f"   Block number: {w3.eth.block_number}")

# Get deployer account
deployer = w3.eth.accounts[0]
print(f"\n👤 Deployer: {deployer}")
balance = w3.eth.get_balance(deployer)
print(f"   Balance: {w3.from_wei(balance, 'ether')} ETH")

# Read contract source
contract_path = Path("topologyB_docker/hardhat/contracts/FederatedLearningAggregator.sol")
if not contract_path.exists():
    print(f"❌ Contract not found at {contract_path}")
    exit(1)

print(f"\n📄 Reading contract from {contract_path}...")
with open(contract_path) as f:
    contract_source = f.read()

# Compile contract
print("🔧 Compiling contract...")
try:
    compiled_sol = compile_source(
        contract_source,
        output_values=['abi', 'bin']
    )
    
    # Get contract interface
    contract_id, contract_interface = compiled_sol.popitem()
    bytecode = contract_interface['bin']
    abi = contract_interface['abi']
    
    print(f"   ✅ Compiled successfully")
    print(f"   Bytecode size: {len(bytecode)} bytes")
    print(f"   ABI methods: {len(abi)}")
    
except Exception as e:
    print(f"   ❌ Compilation failed: {e}")
    exit(1)

# Deploy contract
print("\n🚀 Deploying contract...")
Contract = w3.eth.contract(abi=abi, bytecode=bytecode)

try:
    # Estimate gas
    gas_estimate = Contract.constructor().estimate_gas({'from': deployer})
    print(f"   Gas estimate: {gas_estimate}")
    
    # Deploy
    tx_hash = Contract.constructor().transact({
        'from': deployer,
        'gas': gas_estimate + 100000
    })
    print(f"   TX submitted: {tx_hash.hex()}")
    
    # Wait for receipt
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    contract_address = tx_receipt['contractAddress']
    
    print(f"\n✅ Contract deployed successfully!")
    print(f"   Address: {contract_address}")
    print(f"   Block: {tx_receipt['blockNumber']}")
    print(f"   Gas used: {tx_receipt['gasUsed']}")
    
except Exception as e:
    print(f"   ❌ Deployment failed: {e}")
    exit(1)

# Save deployment info
deployment = {
    "contractAddress": contract_address,
    "deployer": deployer,
    "network": "localhost",
    "chainId": w3.eth.chain_id,
    "blockNumber": tx_receipt['blockNumber'],
    "gasUsed": tx_receipt['gasUsed'],
    "timestamp": w3.eth.get_block(tx_receipt['blockNumber'])['timestamp']
}

deployment_file = Path("deployment.json")
with open(deployment_file, 'w') as f:
    json.dump(deployment, f, indent=2)
print(f"\n💾 Deployment saved to {deployment_file}")

# Save ABI
abi_file = Path("contract_abi.json")
with open(abi_file, 'w') as f:
    json.dump(abi, f, indent=2)
print(f"💾 ABI saved to {abi_file}")

# Test contract
print("\n🧪 Testing contract...")
contract = w3.eth.contract(address=contract_address, abi=abi)

try:
    # Check coordinator
    coordinator = contract.functions.coordinator().call()
    print(f"   Coordinator: {coordinator}")
    assert coordinator == deployer, "Coordinator mismatch!"
    
    # Check current round
    current_round = contract.functions.currentRound().call()
    print(f"   Current round: {current_round}")
    
    print("\n✅ Contract is functional!")
    
except Exception as e:
    print(f"   ❌ Contract test failed: {e}")

print("\n" + "=" * 60)
print("✅ DEPLOYMENT COMPLETE")
print("=" * 60)
print(f"\nContract Address: {contract_address}")
print(f"RPC URL: {RPC_URL}")
print("\nYou can now use this contract in your experiments.")
