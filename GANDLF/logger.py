#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:55:44 2021

@author: siddhesh
"""

import os
from typing import Dict, Union
import torch


class Logger:
    def __init__(
        self, logger_csv_filename: str, metrics: Dict[str, float]
    ) -> None:
        """
        Logger class to log the training and validation metrics to a csv file.

        Args:
            logger_csv_filename (str): Path to a filename where the csv has to be stored.
            metrics (Dict[str, float]): The metrics to be logged.
        """
        self.filename = logger_csv_filename
        self.metrics = metrics

    def write_header(self, mode="train"):
        self.csv = open(self.filename, "a")
        if os.stat(self.filename).st_size == 0:
            mode_lower = mode.lower()
            row = "epoch_no," + mode_lower + "_loss,"
            row += (
                ",".join(
                    [mode_lower + "_" + metric for metric in self.metrics]
                )
                + ","
            )
            row = row[:-1]
            row += "\n"
            self.csv.write(row)
        # else:
        #     print("Found a pre-existing file for logging, now appending logs to that file!")
        self.csv.close()

    def write(
        self, epoch_number: int, loss: float, epoch_metrics: Dict[str, float]
    ) -> None:
        """
        Write the epoch number, loss and metrics to the csv file.

        Args:
            epoch_number (int): The epoch number.
            loss (float): The loss value.
            epoch_metrics (Dict[str, float]): The metrics to be logged.
        """
        self.csv = open(self.filename, "a")
        row = ""
        row += str(epoch_number) + ","
        if torch.is_tensor(loss):
            row += str(loss.cpu().item())
        else:
            row += str(loss)
        row += ","

        for metric in epoch_metrics:
            if torch.is_tensor(epoch_metrics[metric]):
                row += str(epoch_metrics[metric].cpu().item())
            else:
                row += str(epoch_metrics[metric])
            row += ","
        row = row[:-1]
        self.csv.write(row)
        self.csv.write("\n")
        self.csv.close()

    def close(self):
        self.csv.close()


class LoggerGAN:
    def __init__(self, logger_csv_filename, metrics):
        self.filename = logger_csv_filename
        self.metrics = metrics

    def write_header(self, mode: str = "train") -> None:
        """Create a header for the csv file when training GAN networks
        Parameters:
            mode (str): "train" or "valid" or "test"
        """
        with open(self.filename, "a") as log_file:
            if os.stat(self.filename).st_size == 0:
                mode_lower = mode.lower()
                if mode_lower == "train":
                    row = (
                        "epoch_no,"
                        + mode_lower
                        + "_disc_loss,"
                        + mode_lower
                        + "_gen_loss,"
                    )
                else:
                    row = (
                        "epoch_no,"
                        + mode_lower
                        + "_disc_fake_loss,"
                        + mode_lower
                        + "_disc_real_loss,"
                    )
                row += (
                    ",".join(
                        [mode_lower + "_" + metric for metric in self.metrics]
                    )
                    + ","
                )
                row = row[:-1]
                row += "\n"
                log_file.write(row)

    @staticmethod
    def _parse_input_value(numeric_input: Union[float, torch.Tensor]) -> str:
        """Parse the input value to a string
        Parameters:
            numeric_input (float or torch.Tensor): The input value
        Returns:
            str: The parsed input value
        """
        if torch.is_tensor(numeric_input):
            return str(numeric_input.cpu().item())  #
        else:
            return str(numeric_input)

    def write(
        self,
        epoch_number: int,
        disc_loss: Union[float, torch.Tensor],
        gen_loss: Union[float, torch.Tensor],
        epoch_metrics: Dict[str, Union[float, torch.Tensor]],
    ) -> None:
        """
        Write the metrics to the csv file.
        Parameters:
            epoch_number (int): The epoch number
            disc_loss (float or torch.Tensor): The discriminator loss
            gen_loss (float or torch.Tensor): The generator loss
            epoch_metrics (dict): The epoch metrics
        """
        with open(self.filename, "a") as log_file:
            row = ""
            row += self._parse_input_value(epoch_number) + ","
            row += self._parse_input_value(disc_loss) + ","
            row += self._parse_input_value(gen_loss) + ","

            for metric in epoch_metrics:
                row += self._parse_input_value(epoch_metrics[metric]) + ","
            row = row[:-1]  # ??
            log_file.write(row)
            log_file.write("\n")
