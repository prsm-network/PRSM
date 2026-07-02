// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

/// @notice Minimal ProvenanceRegistry for ContentAccessVerifier tests — set a content's
///         creator + rate, then read it back via getCreatorAndRate.
contract MockProvenanceRegistry {
    mapping(bytes32 => address) public creatorOf;
    mapping(bytes32 => uint16) public rateOf;

    function setCreator(bytes32 contentHash, address creator, uint16 rateBps) external {
        creatorOf[contentHash] = creator;
        rateOf[contentHash] = rateBps;
    }

    function getCreatorAndRate(bytes32 contentHash)
        external
        view
        returns (address, uint16)
    {
        return (creatorOf[contentHash], rateOf[contentHash]);
    }
}
