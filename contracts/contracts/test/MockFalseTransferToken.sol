// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

/// @dev Test-only. A token that SILENTLY returns false from transferFrom (never reverts) while
/// reporting ample balance + allowance — i.e. the "non-reverting ERC20" a bridge must defend
/// against. Used to prove FTNSBridge.bridgeOut (sp1433) reverts TransferFailed instead of
/// proceeding to burn/mint as if it had received the tokens (mint-from-nothing fund loss).
contract MockFalseTransferToken {
    function balanceOf(address) external pure returns (uint256) {
        return type(uint256).max;
    }

    function allowance(address, address) external pure returns (uint256) {
        return type(uint256).max;
    }

    function transferFrom(address, address, uint256) external pure returns (bool) {
        return false;
    }
}
