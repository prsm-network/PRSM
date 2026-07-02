// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/// @notice Minimal ProvenanceRegistry surface — the authoritative creator for content, so the
///         access fee cannot be redirected to a party the payer chooses. Mirrors the interface
///         RoyaltyDistributor already consumes.
interface IProvenanceRegistry {
    function getCreatorAndRate(bytes32 contentHash)
        external
        view
        returns (address creator, uint16 rateBps);
}

/**
 * @title ContentAccessVerifier
 * @notice The production ``IRoyaltyPaymentVerifier`` for KeyDistribution (Tier B/C paid-decrypt,
 *         sprint 1353). A consumer pays the release fee for a piece of content here; that records
 *         the payment so ``KeyDistribution.release`` can confirm it, and credits the fee to the
 *         content's REGISTERED creator (looked up on-chain — never chosen by the payer).
 *
 * Flow:
 *   1. Publisher deposits the wrapped content-key to KeyDistribution, naming THIS contract as the
 *      ``royalty`` verifier and a ``releaseFeeFtnsWei``.
 *   2. Consumer ``approve``s FTNS to this contract, then calls
 *      ``payForAccess(contentHash, feeWei)`` — the fee is pulled, credited to the creator's
 *      claimable pool, and the (payer, contentHash, feeWei) payment is recorded.
 *   3. Consumer (or anyone) calls ``KeyDistribution.release(contentHash, consumer)``; the contract
 *      calls ``verifyPayment`` here → true → the key is released.
 *   4. Creator withdraws accumulated fees via ``claim()``.
 *
 * Design notes:
 *   - PULL payment (claimable pool), mirroring RoyaltyDistributor — safe against a creator address
 *     that reverts on ERC-20 receive (a push would let such a creator brick every buyer).
 *   - The fee routes 100% to the creator. The three-way RoyaltyDistributor split is NOT used: its
 *     ``distributeRoyalty`` requires a non-zero serving node, which a direct key-release access
 *     payment has no notion of. A treasury/serving split can be layered later if desired.
 *   - ``verifyPayment`` stays true once paid (persistent) — a paying licensee can have the key
 *     re-released. The (payer, contentHash, feeWei) tuple must match EXACTLY, so a consumer must
 *     pay precisely the fee the publisher set on the deposit.
 */
contract ContentAccessVerifier is ReentrancyGuard {
    IERC20 public immutable ftns;
    IProvenanceRegistry public immutable registry;

    /// keccak256(payer, contentHash, feeWei) => paid
    mapping(bytes32 => bool) public paid;
    /// creator => withdrawable FTNS
    mapping(address => uint256) public claimable;
    uint256 public totalClaimable;

    event AccessPaid(
        address indexed payer,
        bytes32 indexed contentHash,
        uint256 feeWei,
        address indexed creator
    );
    event RoyaltyClaimed(address indexed creator, uint256 amount);

    error ZeroFee();
    error ContentNotRegistered(bytes32 contentHash);
    error PullFailed();
    error NothingToClaim();
    error TransferFailed();

    constructor(address _ftns, address _registry) {
        require(_ftns != address(0) && _registry != address(0), "zero address");
        ftns = IERC20(_ftns);
        registry = IProvenanceRegistry(_registry);
    }

    function _key(address payer, bytes32 contentHash, uint256 feeWei)
        internal
        pure
        returns (bytes32)
    {
        return keccak256(abi.encodePacked(payer, contentHash, feeWei));
    }

    /// @notice The IRoyaltyPaymentVerifier surface KeyDistribution.release() calls.
    function verifyPayment(address payer, bytes32 contentHash, uint256 feeWei)
        external
        view
        returns (bool)
    {
        return paid[_key(payer, contentHash, feeWei)];
    }

    /// @notice Pay the release fee for ``contentHash``. Records the payment (so the key can be
    ///         released to ``msg.sender``) and credits the fee to the content's registered creator.
    function payForAccess(bytes32 contentHash, uint256 feeWei) external nonReentrant {
        if (feeWei == 0) revert ZeroFee();
        // sp1356 (review F8/F11): idempotent — a repeat payment for the same
        // (payer, content, fee) is a NO-OP, never a second charge. release is permissionless and
        // verifyPayment is persistent, so once paid the key is releasable forever; charging again
        // buys nothing. A consumer retry (RPC lag, unsaved plaintext) must not double-spend.
        if (paid[_key(msg.sender, contentHash, feeWei)]) return;
        (address creator, ) = registry.getCreatorAndRate(contentHash);
        if (creator == address(0)) revert ContentNotRegistered(contentHash);

        // Pull the fee before recording (reverting pull ⇒ no state change).
        if (!ftns.transferFrom(msg.sender, address(this), feeWei)) revert PullFailed();

        paid[_key(msg.sender, contentHash, feeWei)] = true;
        claimable[creator] += feeWei;
        totalClaimable += feeWei;

        emit AccessPaid(msg.sender, contentHash, feeWei, creator);
    }

    /// @notice Creator withdraws accumulated access fees (pull payment).
    function claim() external nonReentrant {
        uint256 amt = claimable[msg.sender];
        if (amt == 0) revert NothingToClaim();
        claimable[msg.sender] = 0;
        totalClaimable -= amt;
        if (!ftns.transfer(msg.sender, amt)) revert TransferFailed();
        emit RoyaltyClaimed(msg.sender, amt);
    }
}
